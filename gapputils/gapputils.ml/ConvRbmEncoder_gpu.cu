/*
 * ConvRbmEncoder_gpu.cu
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT
#include "ConvRbmEncoder.h"

#include <capputils/Verifier.h>

#include <tbblas/plus.hpp>
#include <tbblas/conv.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/random.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/real.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/math.hpp>
#include <tbblas/expand.hpp>

#include <curand.h>

#include "RbmModel.h"

#include <capputils/Logbook.h>

namespace gapputils {

namespace ml {

template<class T>
struct softmax : thrust::binary_function<T, unsigned, T> {

  softmax(unsigned width, unsigned blockSize) : width(width), blockSize(blockSize) { }

  __host__ __device__
  T operator()(const T& value, const unsigned& idx) const {
    T res = 0;
    const int offset = (idx % width) % blockSize + ((idx / width) % blockSize) * width;
    for (unsigned j = 0; j < blockSize; ++j)
      for (unsigned i = 0; i < blockSize; ++i)
        res += exp(*(&value + i + j * width - offset));
    return exp(value) / (1 + res);
  }

private:
  unsigned blockSize, width;
};

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl
#define LOCATE2(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

typedef tbblas::tensor<tbblas::complex<double>, 3, true> ctensor_t;

void ConvRbmEncoder::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace capputils;
  using namespace thrust::placeholders;
  using namespace tbblas;
  
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);
  
  /*ConvRbmEncoder test;
  std::cout << std::endl << "ConvRbmEncoder (device): " << sizeof(test) << std::endl;
  LOCATE(test, Model);
  LOCATE(test, Inputs);
  LOCATE(test, Outputs);
  LOCATE(test, Direction);
  LOCATE(test, Sampling);
  LOCATE(test, Pooling);
  LOCATE(test, Auto);
  LOCATE(test, OutputDimension);
  LOCATE2(test, data);*/

  if (!getModel()) {
    dlog(Severity::Warning) << "No model given. Aborting!";
    return;
  }

  if (!getInputs() || getInputs()->size() == 0) {
    dlog(Severity::Warning) << "No input data given. Aborting!";
    return;
  }

  dlog() << "Encoding tensors ...";

  curandGenerator_t gen;
  curandStatus_t status;
  if ((status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)) != CURAND_STATUS_SUCCESS) {
    dlog(Severity::Warning) << "Could not create random number generator: " << status;
    return;
  }

  const unsigned sampleCount = getInputs()->size();

  boost::shared_ptr<ConvRbmModel> crbm = getModel();

  const unsigned dimCount = ConvRbmModel::dimCount;
  const unsigned filterCount = crbm->getFilters()->size();
  const unsigned blockSize = crbm->getPoolingBlockSize();
  const host_tensor_t::dim_t& filterDim = crbm->getFilters()->at(0)->size();
  const host_tensor_t::dim_t& inputDim = getInputs()->at(0)->size();
  int poolingChannels = 1;
  if (getPooling() == PoolingMethod::PositionalMaxPooling)
    poolingChannels = 3;
  else if (getPooling() == PoolingMethod::StackPooling)
    poolingChannels = blockSize * blockSize;

  host_tensor_t::dim_t layerDim, visibleDim, hiddenDim, paddedDim, start;

  if (getDirection() == CodingDirection::Decode && getPooling())
    assert((inputDim[dimCount - 1] % poolingChannels) == 0);

  int filterWeightCount = 1, layerVoxelCount = 1, visibleVoxelCount = 1;
  for (unsigned i = 0; i < dimCount; ++i) {

    if (getDirection() == CodingDirection::Encode) {
      visibleDim[i] = inputDim[i];
      layerDim[i] = inputDim[i] - filterDim[i] + 1;
      hiddenDim[i] = (i < dimCount - 1 ? layerDim[i] : filterCount);
    } else {
      if (getPooling()) {
        hiddenDim[i] = (i < dimCount - 1 ? inputDim[i] * blockSize : inputDim[i] / poolingChannels);
      } else {
        hiddenDim[i] = inputDim[i];
      }
      layerDim[i] = (i < dimCount - 1 ? hiddenDim[i] : 1);
      visibleDim[i] = layerDim[i] + filterDim[i] - 1;
    }
    
    paddedDim[i] = visibleDim[i] + filterDim[i] - 1;
    start[i] = filterDim[i] - 1;
    filterWeightCount *= filterDim[i];
    layerVoxelCount *= layerDim[i];
    visibleVoxelCount *= visibleDim[i];
  }
  
  std::vector<int> outputDim(3);
  if (getDirection() == CodingDirection::Encode) {
    if (getPooling()) {
      outputDim[0] = hiddenDim[0] / blockSize;
      outputDim[1] = hiddenDim[1] / blockSize;
      outputDim[2] = hiddenDim[2] * poolingChannels;
    } else {
      outputDim[0] = hiddenDim[0];
      outputDim[1] = hiddenDim[1];
      outputDim[2] = hiddenDim[2];
    }
  } else {
    outputDim[0] = visibleDim[0];
    outputDim[1] = visibleDim[1];
    outputDim[2] = visibleDim[2];
  }
  newState->setOutputDimension(outputDim);

  //if (getDirection() == CodingDirection::Encode) {
    assert((layerDim[0] % blockSize) == 0);
    assert((layerDim[1] % blockSize) == 0);
  //}
  assert((layerVoxelCount % 2) == 0);
  assert((visibleVoxelCount % 2) == 0);

  // Encode the tensors
  std::vector<boost::shared_ptr<host_tensor_t> >& X = *getInputs();
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > Y(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Copy filters to the device
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<device_tensor_t > F;
  for (unsigned i = 0; i < filters.size(); ++i) {
    device_tensor_t filter(filters[i]->size());
    thrust::copy(filters[i]->begin(), filters[i]->end(), filter.begin());
    F.push_back(filter);
  }

  value_t b = crbm->getVisibleBias();
  std::vector<value_t>& c = *crbm->getHiddenBiases();

  device_tensor_t v(visibleDim), vneg(visibleDim), vtemp(visibleDim), padded(paddedDim);
  device_tensor_t poshidprobs(layerDim), poshidstates(layerDim);

  for (int iSample = 0; iSample < sampleCount && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {

    if (getDirection() == CodingDirection::Encode) {

      /*** START POSITIVE PHASE ***/

      // Get and normalize current sample
      thrust::copy(X[iSample]->begin(), X[iSample]->end(), v.begin());
      if (crbm->getIsGaussian()) {
        v = v - crbm->getMean();
        v = v / crbm->getStddev();
      }

      boost::shared_ptr<host_tensor_t> h(new host_tensor_t(hiddenDim));

      // For each filter (Could be written as a single 4D convolution in case of a 2D image and 3D filter))
      for (unsigned k = 0; k < filterCount; ++k) {

        // Calculate p(h_k | v, F) = sigm((~F_k * v) + c_k)
        poshidstates = tbblas::conv(tbblas::flip(F[k]), v);
        poshidstates = poshidstates + c[k];               // x = ~F_k * v + c_k

        // I'm using the state array here for the sum. Not nice but works fine and saves some space
        thrust::transform(poshidstates.data().begin(), poshidstates.data().end(),
            thrust::make_counting_iterator(0), poshidprobs.data().begin(),
            softmax<value_t>(layerDim[0], blockSize));

        if (getSampling()) {
          // fill states with random numbers which are then used to sample the units
          // TODO: use curandGenerateUniform if value_t == float
          if ((status = curandGenerateUniformDouble(gen,
              poshidstates.data().data().get(),
              poshidstates.data().size())) != CURAND_STATUS_SUCCESS)
          {
            dlog(Severity::Error) << "Could not generate random numbers: " << status;
            return;
          }

          // Sample the hidden states
          thrust::transform(
              poshidprobs.data().begin(), poshidprobs.data().end(), poshidstates.data().begin(),
              poshidstates.data().begin(), _1 > _2
          );

          thrust::copy(poshidstates.data().begin(), poshidstates.data().end(),
              h->data().begin() + k * layerVoxelCount);
        } else {
          thrust::copy(poshidprobs.data().begin(), poshidprobs.data().end(),
              h->data().begin() + k * layerVoxelCount);
        }
      } // for filters

      if (getPooling()) {
        host_tensor_t& input = *h;
        boost::shared_ptr<host_tensor_t> output(new host_tensor_t(hiddenDim[0] / blockSize,
            hiddenDim[1] / blockSize, hiddenDim[2] * poolingChannels));
        const host_tensor_t::dim_t& size2 = output->size();

        for (unsigned z = 0; z < hiddenDim[2]; ++z) {
          for (unsigned y = 0; y < hiddenDim[1]; y += blockSize) {
            for (unsigned x = 0; x < hiddenDim[0]; x += blockSize) {
              unsigned xmax = 0, ymax = 0;
              host_tensor_t::value_t poolingValue = 0;
              if (getPooling() == PoolingMethod::MaxPooling || getPooling() == PoolingMethod::PositionalMaxPooling)
                poolingValue = input.data()[(z * hiddenDim[1] + y) * hiddenDim[0] + x];
              for (int dy = 0, dz = 0; dy < blockSize; ++dy) {
                for (int dx = 0; dx < blockSize; ++dx, ++dz) {
                  host_tensor_t::value_t value = input.data()[(z * hiddenDim[1] + y + dy) * hiddenDim[0] + x + dx];
                  switch (getPooling()) {
                  case PoolingMethod::AvgPooling:
                    poolingValue += value;
                    break;

                  case PoolingMethod::MaxPooling:
                  case PoolingMethod::PositionalMaxPooling:
                    if (value > poolingValue) {
                      poolingValue = value;
                      xmax = dx;
                      ymax = dy;
                    }
                    break;

                  case PoolingMethod::StackPooling:
                    output->data()[((z * poolingChannels + dz) * size2[1] + y / blockSize) * size2[0] + x / blockSize] = value;
                    break;
                  }

                }
              }
              if (getPooling() == PoolingMethod::AvgPooling)
                poolingValue /= (blockSize * blockSize);

              switch (getPooling()) {
              case PoolingMethod::PositionalMaxPooling:
                output->data()[((z * poolingChannels + 1) * size2[1] + y / blockSize) * size2[0] + x / blockSize] =
                    (host_tensor_t::value_t)xmax / (host_tensor_t::value_t)(blockSize - 1);
                output->data()[((z * poolingChannels + 2) * size2[1] + y / blockSize) * size2[0] + x / blockSize] =
                    (host_tensor_t::value_t)ymax / (host_tensor_t::value_t)(blockSize - 1);

              case PoolingMethod::AvgPooling: case PoolingMethod::MaxPooling:
                  output->data()[(z * poolingChannels * size2[1] + y / blockSize) * size2[0] + x / blockSize] = poolingValue;
                  break;
              }
            }
          }
        }
        Y->push_back(output);
      } else {
        Y->push_back(h);
      }
    } else { /*** DECODING ***/
      host_tensor_t unpooled(hiddenDim);
//      dlog(Severity::Message) << "Filter " << iSample + 1;
      if (getPooling()) {
        host_tensor_t& input = *X[iSample];

//        std::cout << tbblas::dot(input, input) << ", " << std::flush;

        for (unsigned z = 0; z < inputDim[2]; z += poolingChannels) {
          for (unsigned y = 0; y < inputDim[1]; ++y) {
            for (unsigned x = 0; x < inputDim[0]; ++x) {
              host_tensor_t::value_t poolingValue = input.data()[(z * inputDim[1] + y) * inputDim[0] + x];
              unsigned xmax = 0;
              unsigned ymax = 0;

              switch (getPooling()) {
              case PoolingMethod::PositionalMaxPooling:
                xmax = input.data()[((z + 1) * inputDim[1] + y) * inputDim[0] + x] * (double)(blockSize - 1) + 0.5;
                ymax = input.data()[((z + 2) * inputDim[1] + y) * inputDim[0] + x] * (double)(blockSize - 1) + 0.5;
                xmax = max(0, min(blockSize - 1, xmax));
                ymax = max(0, min(blockSize - 1, ymax));

              case PoolingMethod::MaxPooling:
                unpooled.data()[(z / poolingChannels * hiddenDim[1] + y * blockSize + ymax) * hiddenDim[0] + x * blockSize + xmax] = poolingValue;
                break;

              case PoolingMethod::AvgPooling:
                for (int dy = 0; dy < blockSize; ++dy) {
                  for (int dx = 0; dx < blockSize; ++dx) {
                    unpooled.data()[(z / poolingChannels * hiddenDim[1] + y * blockSize + dy) * hiddenDim[0] + x * blockSize + dx] = poolingValue;
                  }
                }
                break;

              case PoolingMethod::StackPooling:
                for (int dy = 0, dz = 0; dy < blockSize; ++dy) {
                  for (int dx = 0; dx < blockSize; ++dx, ++dz) {
                    unpooled.data()[(z / poolingChannels * hiddenDim[1] + y * blockSize + dy) * hiddenDim[0] + x * blockSize + dx] =
                        input.data()[((z + dz) * inputDim[1] + y) * inputDim[0] + x];
                  }
                }
                break;
              }
            }
          }
        }
      } else {
        unpooled = *X[iSample];
      }

      /*** START NEGATIVE PHASE ***/

      // Calculate p(v | H, F) = sigm(sum(W_k * h_k) + b)
      thrust::fill(vneg.data().begin(), vneg.data().end(), value_t(0));
      if (getSingleFilter() < 0) {
        for (unsigned k = 0; k < filterCount; ++k) {
          thrust::copy(unpooled.data().begin() + k * layerVoxelCount,
                unpooled.data().begin() + (k + 1) * layerVoxelCount,
                padded[start, layerDim].begin());

          vtemp = tbblas::conv(padded,F[k]);
          vneg = vneg + vtemp;
        }
      } else {
        const unsigned k = getSingleFilter();
        thrust::copy(unpooled.data().begin() + k * layerVoxelCount,
              unpooled.data().begin() + (k + 1) * layerVoxelCount,
              padded[start, layerDim].begin());

        vtemp = tbblas::conv(padded,F[k]);
        vneg = vneg + vtemp;
      }
      vneg = vneg + b;

      // For the binary case
      if (!crbm->getIsGaussian()) {
        thrust::transform(vneg.begin(), vneg.end(), vneg.begin(),
            sigmoid<value_t>());

        if (getSampling()) {
          if ((status = curandGenerateUniformDouble(gen, vtemp.data().data().get(), vtemp.data().size())) != CURAND_STATUS_SUCCESS)
          {
            dlog(Severity::Error) << "Could not generate random numbers: " << status;
            return;
          }

          thrust::transform(
              vneg.data().begin(), vneg.data().end(), vtemp.data().begin(),
              vneg.data().begin(), _1 > _2
          );
        }
      } else {
        if (getSampling()) {
          if ((status = curandGenerateNormalDouble(gen,
              vtemp.data().data().get(),
              vtemp.data().size(),
              0, 1.0)) != CURAND_STATUS_SUCCESS)
          {
            dlog(Severity::Error) << "Could not generate random numbers: " << status;
            return;
          }

          thrust::transform(
              vneg.data().begin(), vneg.data().end(), vtemp.data().begin(),
              vneg.data().begin(), thrust::plus<value_t>()
          );
        }
        value_t mean = crbm->getMean();
        value_t stddev = crbm->getStddev();

//        dlog() << "Mean = " << mean << "; Stddev = " << stddev;
        vneg = vneg * stddev;
//        dlog() << "vneg * stddev: " << vneg[seq(0,0,0)] << ", " << vneg[seq(1,0,0)] << ", " << vneg[seq(2,0,0)];
        vneg = vneg + mean;
//        dlog() << "vneg + mean: " << vneg[seq(0,0,0)] << ", " << vneg[seq(1,0,0)] << ", " << vneg[seq(2,0,0)];
      }
//      dlog(Severity::Message) << "Result: " << vneg[seq(0,0,0)] << ", " << vneg[seq(1,0,0)] << ", " << vneg[seq(2,0,0)];
      Y->push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(vneg)));
    }
    if (monitor) monitor->reportProgress(iSample * 100 / X.size());
  } // for samples

  if ((status = curandDestroyGenerator(gen)) != CURAND_STATUS_SUCCESS)
  {
    dlog(Severity::Error) << "Could not destroy random number generator: " << status;
    return;
  }
  dlog() << "Finished";

  newState->setOutputs(Y);
}

}

}
