/*
 * conv_rbm.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef GML_DBN_CONV_RBM_HPP_
#define GML_DBN_CONV_RBM_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/dot.hpp>

#include <omp.h>

#include "math.hpp"
#include "mult_sum.hpp"
#include "repeat_mult.hpp"
#include "repeat_mult_sum.hpp"
#include "convolution_type.hpp"
#include "unit_type.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>


namespace gml {

namespace dbn {

/// This class creates multiple threads
/**
 * Some changes to the previous design:
 * - No thread local variables. Thread local variables are replaced by vectors of
 *   shared pointers. Better control over the creation and destruction of variables.
 * - Every thread has a local reference to the memory. Makes code cleaner.
 */

template<class T, unsigned dims>
class conv_rbm {
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;
  typedef tbblas::complex<value_t> complex_t;
  typedef tbblas::fft_plan<dimCount> plan_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef tbblas::tensor<complex_t, dimCount, true> ctensor_t;
  typedef tbblas::tensor<complex_t, dimCount, false> host_ctensor_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  host_tensor_t _visibleBiases, _mask;
  v_host_tensor_t _filters, _hiddenBiases;
  dim_t _filterKernelSize;

  unit_type _visibleUnitType, _hiddenUnitType;
  convolution_type _convolutionType;

  value_t _mean, _stddev;

  // weights and bias terms in GPU memory
  tensor_t b;

  std::vector<boost::shared_ptr<ctensor_t> > cF, cc;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t visible_size, hidden_size, size,
        visible_layer_size, hidden_layer_size, layer_size,
        filter_batch_size, layer_batch_size, hidden_topleft;

  // one element per thread
  std::vector<boost::shared_ptr<tensor_t> > v_v, v_h, v_v_mask, v_h_mask;
  std::vector<boost::shared_ptr<ctensor_t> > v_cv, v_ch_full, v_ch;
  std::vector<boost::shared_ptr<plan_t> > v_plan_v, v_iplan_v, v_plan_h, v_iplan_h;
  tensor_t _hiddens;

  int _gpu_count, _device_count, _filter_batch_length;
  bool _memory_allocated, _double_weights, _padding, _host_updated;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm(size_t gpu_count = 1) : _mean(0), _stddev(1),
    _gpu_count(gpu_count), _filter_batch_length(1),
    _memory_allocated(false), _double_weights(false), _padding(false), _host_updated(true)
  {
    assert(_gpu_count > 0);

    v_v.resize(gpu_count);
    v_h.resize(gpu_count);
    v_v_mask.resize(gpu_count);
    v_h_mask.resize(gpu_count);
    v_cv.resize(gpu_count);
    v_ch_full.resize(gpu_count);
    v_ch.resize(gpu_count);

    v_plan_v.resize(gpu_count);
    v_iplan_v.resize(gpu_count);
    v_plan_h.resize(gpu_count);
    v_iplan_h.resize(gpu_count);
  }

private:
  conv_rbm(const conv_rbm&);

public:
  // Automatically frees GPU memory (called from non-parallel code, throws an exception if non-freed memory from multiple threads remains)
  virtual ~conv_rbm() {
    if (_memory_allocated)
      free_gpu_memory();
  }

  // This functions can run in parallel. They also create threads

  /// Transforms
  void allocate_gpu_memory() {
    using namespace tbblas;

    // TODO: check if peer access is enabled

    if (_memory_allocated)
      return;

    _memory_allocated = true;

    setup_threads();

    // Prepare sizes
    visible_size = _visibleBiases.size();
    size = visible_size;

    if (_padding) {
      for (size_t j = 0; j < dimCount - 1; ++j)
        size[j] = upper_power_of_two(size[j]);
    }

    visible_layer_size = visible_size;
    layer_size = filter_batch_size = layer_batch_size = size;
    visible_layer_size[dimCount - 1] = layer_size[dimCount - 1] = 1;
    filter_batch_size[dimCount - 1] = size[dimCount - 1] * _filter_batch_length;
    layer_batch_size[dimCount - 1] = _filter_batch_length;

    if (_convolutionType == convolution_type::Valid){
      hidden_topleft = _filterKernelSize / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq(0,0,0,0);
    }
    hidden_layer_size = visible_layer_size - 2 * hidden_topleft;
    hidden_size = visible_size - 2 * hidden_topleft;
    hidden_size[dimCount - 1] = _filters.size();

    _hiddens = zeros<value_t>(hidden_size);

    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, 3), cB = cA;

      // throw exception instead
      assert(dot(A - B, A - B) == 0);

      A = ifft(cA, 3);
      assert (abs(dot(cA - cB, cA - cB)) == 0);
    }

    b = _visibleBiases;
    cF.resize(_filters.size());
    cc.resize(_hiddenBiases.size());

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      v_v[tid] = boost::make_shared<tensor_t>();
      v_h[tid] = boost::make_shared<tensor_t>();
      v_cv[tid] = boost::make_shared<ctensor_t>();
      v_ch_full[tid] = boost::make_shared<ctensor_t>();
      v_ch[tid] = boost::make_shared<ctensor_t>();


      plan_t& plan_v = *(v_plan_v[tid] = boost::make_shared<plan_t>());
      plan_t& iplan_v = *(v_iplan_v[tid] = boost::make_shared<plan_t>());
      plan_t& plan_h = *(v_plan_h[tid] = boost::make_shared<plan_t>());
      plan_t& iplan_h = *(v_iplan_h[tid] = boost::make_shared<plan_t>());

      tensor_t& v_mask = *(v_v_mask[tid] = boost::make_shared<tensor_t>());
      tensor_t& h_mask = *(v_h_mask[tid] = boost::make_shared<tensor_t>());
      v_mask = zeros<value_t>(layer_size);
      v_mask[seq(0,0,0,0), _mask.size()] = _mask;

      // pad h mask according to convolution shrinkage
      if (_convolutionType == convolution_type::Valid){
        h_mask = zeros<value_t>(layer_size);
        h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
        h_mask = h_mask * v_mask;
      } else {
        h_mask = v_mask;
      }

      // Copy filters to the device and pre-calculate the FFT
      // TODO: Copy filter batches
      {
        tensor_t f, h, kern, pad;
        ctensor_t cf, ch;
        for (size_t k = tid; k < _filters.size(); k += _gpu_count) {
  //        f = *filters[k];

          if (_double_weights)
            kern = 2 * *_filters[k];
          else
            kern = *_filters[k];
          dim_t topleft = size / 2 - kern.size() / 2;
          pad = zeros<value_t>(size);
          pad[topleft, kern.size()] = kern;
          f = ifftshift(pad, dimCount - 1);
          cf = fft(f, dimCount - 1, plan_v);
          cF[k] = boost::make_shared<ctensor_t>(cf);

          h = zeros<value_t>(layer_size);
          h[seq(0,0,0,0), visible_layer_size] = *_hiddenBiases[k];
          ch = fft(h, dimCount - 1, plan_h);
          cc[k] = boost::make_shared<ctensor_t>(ch);
        }
      }
    }
  }

  void free_gpu_memory() {
    _memory_allocated = false;

    setup_threads();

    b = tensor_t();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      v_v[tid] = boost::shared_ptr<tensor_t>();
      v_h[tid] = boost::shared_ptr<tensor_t>();
      v_v_mask[tid] = boost::shared_ptr<tensor_t>();
      v_h_mask[tid] = boost::shared_ptr<tensor_t>();
      v_cv[tid] = boost::shared_ptr<ctensor_t>();
      v_ch_full[tid] = boost::shared_ptr<ctensor_t>();
      v_ch[tid] = boost::shared_ptr<ctensor_t>();

      v_plan_v[tid] = boost::shared_ptr<plan_t>();
      v_iplan_v[tid] = boost::shared_ptr<plan_t>();
      v_plan_h[tid] = boost::shared_ptr<plan_t>();
      v_iplan_h[tid] = boost::shared_ptr<plan_t>();

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        cF[k] = cc[k] = boost::shared_ptr<ctensor_t>();
      }
    }
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    tensor_t& v = *v_v[0];
    tensor_t& v_mask = *v_v_mask[0];
    v = ((v - _mean) / _stddev) * tbblas::repeat(v_mask, size / layer_size);
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    tensor_t& v = *v_v[0];
    tensor_t& v_mask = *v_v_mask[0];
    v = ((v * _stddev) + _mean) * tbblas::repeat(v_mask, size / layer_size);
  }

  void infer_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& v = *v_v[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch_full = *v_ch_full[tid];
      ctensor_t& ch = *v_ch[tid];

      plan_t& plan_v = *v_plan_v[tid];
      plan_t& iplan_v = *v_iplan_v[tid];
      plan_t& plan_h = *v_plan_h[tid];
      plan_t& iplan_h = *v_iplan_h[tid];
      tensor_t& h_mask = *v_h_mask[tid];
      tensor_t& v_mask = *v_v_mask[tid];

      cv = zeros<complex_t>(cF[0]->size(), cF[0]->fullsize());

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        h = zeros<value_t>(layer_size);
        h[hidden_topleft, hidden_layer_size] = _hiddens[seq(0,0,0,(int)k), hidden_layer_size];
        h = h * h_mask;
        ch = fft(h, dimCount - 1, plan_h);

        cv = cv + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier

      #pragma omp critical
      {
        if (tid != 0)
          *v_cv[0] = *v_cv[0] + cv;
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier

      #pragma omp master
      {
        v = ifft(cv, dimCount - 1, iplan_v);

        switch(_visibleUnitType) {
          case unit_type::Bernoulli: v = sigm(v + b); break;
          case unit_type::Gaussian:  v = v + b;       break;
          case unit_type::ReLU:      v = max(0.0, v + b);  break;
          case unit_type::MyReLU:    v = nrelu_mean(v + b); break;
          case unit_type::ReLU1:     v = min(1.0, max(0.0, v + b));  break;
          case unit_type::ReLU2:     v = min(2.0, max(0.0, v + b));  break;
          case unit_type::ReLU4:     v = min(4.0, max(0.0, v + b));  break;
          case unit_type::ReLU8:     v = min(8.0, max(0.0, v + b));  break;
        }
        v = v * repeat(v_mask, size / layer_size);
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier
    }
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& v = *v_v[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch_full = *v_ch_full[tid];
      ctensor_t& ch = *v_ch[tid];
      tensor_t& h_mask = *v_h_mask[tid];

      plan_t& plan_v = *v_plan_v[tid];
      plan_t& iplan_v = *v_iplan_v[tid];
      plan_t& plan_h = *v_plan_h[tid];
      plan_t& iplan_h = *v_iplan_h[tid];

      cudaStreamSynchronize(0);
      #pragma omp barrier

      if (tid != 0)
        cv = *v_cv[0];

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        ch = conj_mult_sum(cv, *cF[k]);

        ch = ch + *cc[k];
        h = ifft(ch, dimCount - 1, iplan_h);

        switch (_hiddenUnitType) {
          case unit_type::Bernoulli: h = sigm(h); break;
          case unit_type::ReLU:      h = max(0.0, h);  break;
          case unit_type::MyReLU:    h = nrelu_mean(h); break;
          case unit_type::ReLU1:     h = min(1.0, max(0.0, h));  break;
          case unit_type::ReLU2:     h = min(2.0, max(0.0, h));  break;
          case unit_type::ReLU4:     h = min(4.0, max(0.0, h));  break;
          case unit_type::ReLU8:     h = min(8.0, max(0.0, h));  break;
        }
        h = h * repeat(h_mask, h.size() / h_mask.size());
        _hiddens[seq(0,0,0,(int)k), hidden_layer_size] = h[hidden_topleft, hidden_layer_size];
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier
    }
  }

  void apply_positive_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    _host_updated = false;
  }

  void apply_negative_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    _host_updated = false;
  }

  // Access to model data
  void set_visibles(const host_tensor_t& v) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    // TODO: how to handle padding?
    *v_v[0] = v;
  }

  void set_visibles(const tensor_t& v) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    // TODO: how to handle padding?
    *v_v[0] = v;
  }

  const tensor_t& visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    // TODO: how to handle padding?
    return *v_v[0];
  }

  void set_hiddens(const host_tensor_t& h) {
    if (!_memory_allocated)
      allocate_gpu_memory();
    _hiddens = h;
  }

  void set_hiddens(const tensor_t& h) {
    if (!_memory_allocated)
      allocate_gpu_memory();
    _hiddens = h;
  }

  const tensor_t& hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return _hiddens;
  }

  v_host_tensor_t& filters() {
    if (!_host_updated)
      write_model_to_host();

    return _filters;
  }

  host_tensor_t& visible_bias() {
    if (!_host_updated)
      write_model_to_host();

    return _visibleBiases;
  }

  v_host_tensor_t& hidden_bias() {
    if (!_host_updated)
      write_model_to_host();

    return _hiddenBiases;
  }

  host_tensor_t& mask() {
    return _mask;
  }

  dim_t& kernel_size() {
    return _filterKernelSize;
  }

  unit_type& visibles_type() {
    return _visibleUnitType;
  }

  unit_type& hiddens_type() {
    return _hiddenUnitType;
  }

  convolution_type& convolution_type() {
    return _convolutionType;
  }

  void set_mean(value_t mean) {
    _mean = mean;
  }

  value_t mean() const {
    return _mean;
  }

  void set_stddev(value_t stddev) {
    _stddev = stddev;
  }

  value_t stddev() const {
    return _stddev;
  }

private:
  void setup_threads() {
    cudaGetDeviceCount(&_device_count);

    assert (_device_count >= _gpu_count);
    assert(omp_get_num_threads() == 1);

    cudaSetDevice(0);
    omp_set_dynamic(0);
    omp_set_num_threads(_gpu_count);
  }

  unsigned int upper_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

  void write_model_to_host() {
    if (_host_updated)
      return;
    _host_updated = true;

    // TODO: write device model to host
  }

};

}

}

#endif /* GML_DBN_CONV_RBM_HPP_ */
