/*
 * RbmDecoder_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include "RbmDecoder.h"

#include <capputils/Verifier.h>

#include "sampling.hpp"

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void RbmDecoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  if (!data)
    data = new RbmDecoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getHiddenVector())
      return;

  RbmModel& rbm = *getRbmModel();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = rbm.getVisibleBiases()->size();
  const unsigned hiddenCount = rbm.getHiddenBiases()->size();
  const unsigned sampleCount = getHiddenVector()->size() / hiddenCount;

//  std::cout << "Decode: " << sampleCount << std::endl;

  ublas::matrix<float> hiddens(sampleCount, hiddenCount);
  std::copy(getHiddenVector()->begin(), getHiddenVector()->end(), hiddens.data().begin());

  tbblas::device_matrix<float> H(sampleCount, hiddenCount);
  H = hiddens;
  tbblas::device_matrix<float>& W = *rbm.getWeightMatrix();
  tbblas::device_vector<float>& b = *rbm.getVisibleBiases();
  tbblas::device_matrix<float> negdata(sampleCount, visibleCount);

  // Calculate p(x | H, W) = sigm(HW' + B)
  negdata = tbblas::prod(H, tbblas::trans(W));
  if (!getUseWeightsOnly()) {
    for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
      tbblas::row(negdata, iRow) += b;
  }

  // For the binary case
  if (!rbm.getIsGaussian())
    thrust::transform(negdata.data().begin(), negdata.data().end(), negdata.data().begin(),
          sigmoid<float>());

  boost::shared_ptr<std::vector<float> > visibleVector(new std::vector<float>(sampleCount * visibleCount));
  if (rbm.getIsGaussian() && !getUseWeightsOnly()) {
    assert (rbm.getVisibleMeans() && rbm.getVisibleMeans()->size());
    assert (rbm.getVisibleStds() && rbm.getVisibleStds()->size());

    const float mean = rbm.getVisibleMeans()->data()[0];
    const float stddev = rbm.getVisibleStds()->data()[0];
    thrust::transform(negdata.data().begin(), negdata.data().end(), negdata.data().begin(), _1 * stddev + mean);
  }

  ublas::matrix<float> visibles(sampleCount, visibleCount);
  visibles = negdata;
  std::copy(visibles.data().begin(), visibles.data().end(), visibleVector->begin());

  // Decode approximation (reconstruction)
  data->setVisibleVector(visibleVector);
}

}

}
