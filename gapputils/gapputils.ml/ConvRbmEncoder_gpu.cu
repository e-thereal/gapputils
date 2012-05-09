/*
 * ConvRbmEncoder_gpu.cu
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT
#include "ConvRbmEncoder.h"

#include <capputils/Verifier.h>

#include <tbblas/tensor_proxy.hpp>
#include <curand.h>

#include "RbmModel.h"

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

void ConvRbmEncoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;
  typedef tbblas::tensor_proxy<host_tensor_t::iterator, 3> host_proxy_t;
  typedef tbblas::tensor_proxy<device_tensor_t::iterator, 3> device_proxy_t;

  if (!data)
    data = new ConvRbmEncoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getModel()) {
    std::cout << "[Warning] No model given. Aborting!" << std::endl;
    return;
  }

  if (!getInputs() || getInputs()->size() == 0) {
    std::cout << "[Warning] No input data given. Aborting!" << std::endl;
    return;
  }

  std::cout << "Encoding tensors ..." << std::flush;

  curandGenerator_t gen;
  curandStatus_t status;
  if ((status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)) != CURAND_STATUS_SUCCESS) {
    std::cout << "[Warning] Could not create random number generator: " << status << std::endl;
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

  assert((layerDim[0] % blockSize) == 0);
  assert((layerDim[1] % blockSize) == 0);
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
        v += -crbm->getMean();
        v = tbblas::copy(v / crbm->getStddev());
      }

      boost::shared_ptr<host_tensor_t> h(new host_tensor_t(hiddenDim));

      // For each filter (Could be written as a single 4D convolution in case of a 2D image and 3D filter))
      for (unsigned k = 0; k < filterCount; ++k) {

        // Calculate p(h_k | v, F) = sigm((~F_k * v) + c_k)
        poshidstates = tbblas::conv(tbblas::flip(F[k]), v, (k ? tbblas::ReuseFT2 : tbblas::ReuseFTNone));
        poshidstates += c[k];               // x = ~F_k * v + c_k

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
            std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
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
//      std::cout << "Filter " << iSample + 1 << ": " << std::flush;
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
        unpooled = tbblas::copy(*X[iSample]);
      }

//      std::cout << tbblas::dot(unpooled, unpooled) << ", " << std::flush;

      /*** START NEGATIVE PHASE ***/

      // Calculate p(v | H, F) = sigm(sum(W_k * h_k) + b)
      thrust::fill(vneg.data().begin(), vneg.data().end(), value_t(0));
      for (unsigned k = 0; k < filterCount; ++k) {
        device_proxy_t paddedProxy = tbblas::subrange(padded, start, layerDim);

        thrust::copy(unpooled.data().begin() + k * layerVoxelCount,
            unpooled.data().begin() + (k + 1) * layerVoxelCount, paddedProxy.begin());

        vtemp = tbblas::conv(F[k], padded);

//        if (k == 0) {
//          std::cout << tbblas::dot(F[k], F[k]) << ", " << std::flush;
//          std::cout << tbblas::dot(padded, padded) << ", " << std::flush;
//        }

        vneg += vtemp;
      }
//      std::cout << tbblas::dot(vneg, vneg) << ", " << std::flush;
      vneg += b;
//      std::cout << tbblas::dot(vneg, vneg) << ", " << std::flush;

      // For the binary case
      if (!crbm->getIsGaussian()) {
        thrust::transform(vneg.begin(), vneg.end(), vneg.begin(),
            sigmoid<value_t>());

        if (getSampling()) {
          if ((status = curandGenerateUniformDouble(gen, vtemp.data().data().get(), vtemp.data().size())) != CURAND_STATUS_SUCCESS)
          {
            std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
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
            std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
            return;
          }

          thrust::transform(
              vneg.data().begin(), vneg.data().end(), vtemp.data().begin(),
              vneg.data().begin(), thrust::plus<value_t>()
          );
        }
        value_t mean = crbm->getMean();
        value_t stddev = crbm->getStddev();

//        std::cout << "[Encoding] Mean = " << mean << "; Stddev = " << stddev << std::endl;
        vneg = tbblas::copy(vneg * stddev);
//        std::cout << tbblas::dot(vneg, vneg) << ", " << std::flush;
        vneg += mean;
//        std::cout << tbblas::dot(vneg, vneg) << ", " << std::flush;
      }
//      std::cout << tbblas::dot(vneg, vneg) << std::endl;
      Y->push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(tbblas::copy(vneg))));
    }
    if (monitor) monitor->reportProgress(iSample * 100 / X.size());
  } // for samples

  if ((status = curandDestroyGenerator(gen)) != CURAND_STATUS_SUCCESS)
  {
    std::cout << "[Error] Could not destroy random number generator: " << status << std::endl;
    return;
  }
  std::cout << " done!" << std::endl;

  data->setOutputs(Y);
}

}

}
