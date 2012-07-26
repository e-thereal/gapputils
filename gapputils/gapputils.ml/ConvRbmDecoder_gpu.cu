/*
 * ConvRbmDecoder_gpu.cu
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "ConvRbmDecoder.h"

#include <capputils/Verifier.h>

#include "RbmModel.h"

#include <curand.h>

#include <tbblas/device_matrix.hpp>

namespace gapputils {

namespace ml {

void ConvRbmDecoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;
  typedef tbblas::tensor_proxy<device_tensor_t::iterator, 3> device_proxy_t;

  if (!data)
    data = new ConvRbmDecoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getModel()) {
    std::cout << "[Warning] No initial model given. Aborting!" << std::endl;
    return;
  }

  if (!getInputs() || getInputs()->size() == 0) {
    std::cout << "[Warning] No data given. Aborting!" << std::endl;
    return;
  }

  std::cout << "Decoding ConvRBM ..." << std::endl;

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
  const host_tensor_t::dim_t& hiddenDim = getInputs()->at(0)->size();
  host_tensor_t::dim_t layerDim, paddedDim, start, visibleDim;

  int filterWeightCount = 1, layerVoxelCount = 1, visibleVoxelCount = 1;
  for (unsigned i = 0; i < dimCount; ++i) {
    layerDim[i] = (i < dimCount - 1 ? hiddenDim[i] : 1);
    visibleDim[i] = layerDim[i] + filterDim[i] - 1;
    paddedDim[i] = visibleDim[i] + filterDim[i] - 1;
    start[i] = filterDim[i] - 1;
    filterWeightCount *= filterDim[i];
    layerVoxelCount *= layerDim[i];
    visibleVoxelCount *= visibleDim[i];
  }

  assert(hiddenDim[dimCount - 1] == filterCount);
  assert((layerDim[0] % blockSize) == 0);
  assert((layerDim[1] % blockSize) == 0);
  assert((layerVoxelCount % 2) == 0);
  assert((visibleVoxelCount % 2) == 0);

  // Train the RBM
  std::vector<boost::shared_ptr<host_tensor_t> >& hiddens = *getInputs();
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > visibles(
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

//  std::cout << "Visible Bias = " << b << std::endl;

  device_tensor_t vneg(visibleDim), vtemp(visibleDim), padded(paddedDim);
  thrust::fill(padded.begin(), padded.end(), value_t(0));
  std::vector<device_tensor_t> poshidstates;
  for (unsigned i = 0; i < filterCount; ++i) {
    poshidstates.push_back(device_tensor_t(layerDim));
  }

  std::cout << "[Info] Starting decoding" << std::endl;
  for (int iSample = 0; iSample < sampleCount; ++iSample) {

    /*** START NEGATIVE PHASE ***/

    // Calculate p(v | H, F) = sigm(sum(W_k * h_k) + b)
    thrust::fill(vneg.data().begin(), vneg.data().end(), value_t(0));
    for (unsigned k = 0; k < filterCount; ++k) {
      device_proxy_t paddedProxy = tbblas::subrange(padded, start, layerDim);

      thrust::copy(hiddens[iSample]->data().begin() + k * layerVoxelCount,
          hiddens[iSample]->data().begin() + (k + 1) * layerVoxelCount, paddedProxy.begin());

      vtemp = tbblas::conv(F[k], padded);
      vneg += vtemp;
    }
    vneg += b;

    // For the binary case
    if (!crbm->getIsGaussian()) {
      thrust::transform(vneg.begin(), vneg.end(), vneg.begin(),
          sigmoid<value_t>());

      if (getSampleVisibles()) {
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
      if (getSampleVisibles()) {
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

//      std::cout << "[Encoding] Mean = " << mean << "; Stddev = " << stddev << std::endl;
      vneg = tbblas::copy(vneg * stddev);
      vneg += mean;
    }

    visibles->push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(tbblas::copy(vneg))));
  }

  if ((status = curandDestroyGenerator(gen)) != CURAND_STATUS_SUCCESS)
  {
    std::cout << "[Error] Could not destroy random number generator: " << status << std::endl;
    return;
  }

  data->setOutputs(visibles);
}

}

}
