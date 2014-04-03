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

#include <omp.h>

#include "math.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>

#include <gml.convrbm4d/UnitType.h>
#include <gml.convrbm4d/ConvolutionType.h>

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
  typedef tbblas::tensor<value_t, dimCount>::dim_t dim_t;
  typedef tbblas::complex<value_t> complex_t;
  typedef tbblas::fft_plan<dimCount> plan_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor<complex_t, dimCount, false> host_ctensor_t;

protected:
  // Model in CPU memory
  boost::shared_ptr<host_tensor_t> _visibleBiases, _mask;
  boost::shared_ptr<v_host_tensor_t> _weights, _hiddenBiases;
  dim_t _filterKernelSize;

  gml::convrbm4d::UnitType _visibleUnitType, _hiddenUnitType;
  gml::convrbm4d::ConvolutionType _convolutionType;

  // weights and bias terms in GPU memory
  tensor_t b;

  std::vector<boost::shared_ptr<ctensor_t> > cF, cc;

  // visible and hidden units in GPU memory
  tensor_t v_master;
  ctensor_t cv_master;

  // one element per thread
  std::vector<boost::shared_ptr<tensor_t> > v_v, v_h;
  std::vector<boost::shared_ptr<ctensor_t> > v_cv, v_ch_full, v_ch;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm() {

  }

  // Automatically frees GPU memory (called from non-parallel code, throws an exception if non-freed memory from multiple threads remains)
  virtual ~conv_rbm() {

  }

  // This functions can run in parallel. Will figure out thread configuration using OpenMP.

  /// Transforms
  void allocate_gpu_memory() {



    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      for (size_t k = tid; k < filters.size(); k += gpuCount) {
//        f = *filters[k];

        if (getDoubleWeights())
          kern = 2 * *filters[k];
        else
          kern = *filters[k];
        dim_t topleft = size / 2 - kern.size() / 2;
        pad = zeros<value_t>(size);
        pad[topleft, kern.size()] = kern;
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_v);
        cF[k] = boost::make_shared<ctensor_t>(cf);

        h = zeros<value_t>(layerSize);
        h[seq(0,0,0,0), originalLayerSize] = *c[k];
        ch = fft(h, dimCount - 1, plan_h);
        cc[k] = boost::make_shared<ctensor_t>(ch);
      }
    }
  }

  void free_gpu_memory() {
    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      cF[k] = cc[k] = boost::shared_ptr<ctensor_t>();
    }
  }

  void infer_visibles() {
    cv = zeros<complex_t>(cF[0]->size(), cF[0]->fullsize());

    dim_t inSize = inputs[i]->size(), inLayerSize = inSize;
    inLayerSize[dimCount - 1] = 1;

    #pragma omp master
    {
      cv_master = zeros<complex_t>(cF[0]->size(), cF[0]->fullsize());
      cudaStreamSynchronize(0);
    }
    #pragma omp barrier

    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      h = zeros<value_t>(layerSize);
      h[topleft, inLayerSize] = (*inputs[i])[seq(0,0,0,(int)k), inLayerSize];
      if (!getOnlyFilters())
        h = h * hMask;
      ch = fft(h, dimCount - 1, plan_h);

      cv = cv + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
    }

    #pragma omp critical
    {
      cv_master = cv_master + cv;
      cudaStreamSynchronize(0);
    }
    #pragma omp barrier

    #pragma omp master
    {
      v = ifft(cv_master, dimCount - 1, iplan_v);

      if (getOnlyFilters()) {
        output = v[seq(0,0,0,0), originalSize];
      } else {
        switch(crbm.getVisibleUnitType()) {
          case UnitType::Bernoulli: v = sigm(v + b); break;
          case UnitType::Gaussian:  v = v + b;       break;
          case UnitType::ReLU:      v = max(0.0, v + b);  break;
          case UnitType::MyReLU:    v = nrelu_mean(v + b); break;
          case UnitType::ReLU1:     v = min(1.0, max(0.0, v + b));  break;
          case UnitType::ReLU2:     v = min(2.0, max(0.0, v + b));  break;
          case UnitType::ReLU4:     v = min(4.0, max(0.0, v + b));  break;
          case UnitType::ReLU8:     v = min(8.0, max(0.0, v + b));  break;
          default:
            dlog(Severity::Warning) << "Unsupported unit type: " << crbm.getVisibleUnitType();
        }
        v = ((v * crbm.getStddev()) + crbm.getMean()) * repeat(vMask, size / layerSize);
        output = v[seq(0,0,0,0), originalSize];
      }

      cudaStreamSynchronize(0);
    }
    #pragma omp barrier
  }

  void infer_hiddens() {
    cudaStreamSynchronize(0);
    #pragma omp barrier

    dim_t outSize = originalSize - 2 * topleft;
    outSize[dimCount - 1] = cF.size();
    dim_t outLayerSize = outSize;
    outLayerSize[dimCount - 1] = 1;

    #pragma omp master
    {
      v_master = zeros<value_t>(size);
      v_master[seq(0,0,0,0), inputs[i]->size()] = *inputs[i];
      if (crbm.getVisibleUnitType() == UnitType::Gaussian)
        v_master = (v_master - crbm.getMean()) / crbm.getStddev();
      v_master = v_master * repeat(vMask, size / layerSize);
      cv_master = fft(v_master, dimCount - 1, plan_v);
      output.resize(outSize, outSize);
      cudaStreamSynchronize(0);
    }
    #pragma omp barrier

    cv = cv_master;

    bool validSize = true;
    for (unsigned j = 0; j < dimCount - 1; ++j) {
      if (v_master.size()[j] != upper_power_of_two(v_master.size()[j])) {
        dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Skipping image!";
        validSize = false;
        break;
      }
    }
    if (!validSize)
      continue;

    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      ch_full = conj(*cF[k]) * cv;
      ch = sum(ch_full, dimCount - 1);
      ch = ch + *cc[k];
      h = ifft(ch, dimCount - 1, iplan_h);

      switch (crbm.getHiddenUnitType()) {
        case UnitType::Bernoulli: h = sigm(h); break;
        case UnitType::ReLU:      h = max(0.0, h);  break;
        case UnitType::MyReLU:    h = nrelu_mean(h); break;
        case UnitType::ReLU1:     h = min(1.0, max(0.0, h));  break;
        case UnitType::ReLU2:     h = min(2.0, max(0.0, h));  break;
        case UnitType::ReLU4:     h = min(4.0, max(0.0, h));  break;
        case UnitType::ReLU8:     h = min(8.0, max(0.0, h));  break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type: " << crbm.getVisibleUnitType();
      }
      h = h * hMask;
      output[seq(0,0,0,(int)k), outLayerSize] = h[topleft, outLayerSize];
    }
    cudaStreamSynchronize(0);
    #pragma omp barrier
  }

  void apply_positive_gradient() {

  }

  void apply_negative_gradient() {

  }

  // Iterators for visible and hidden units per channel
};

}

}

#endif /* GML_DBN_CONV_RBM_HPP_ */
