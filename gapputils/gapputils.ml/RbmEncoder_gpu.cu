/*
 * RbmEncoder_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include "RbmEncoder.h"

#include <capputils/Verifier.h>
#include <algorithm>

#include "sampling.hpp"

#include <iostream>

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

template<class T>
struct min_0 {
__host__ __device__
T  operator()(const T& x) const {
  return max((T)0, x);
}

};

void RbmEncoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  if (!data)
    data = new RbmEncoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getVisibleVector())
    return;

  RbmModel& rbm = *getRbmModel();
  HiddenUnitType hiddenUnitType = rbm.getHiddenUnitType();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = rbm.getVisibleBiases()->size();
  const unsigned hiddenCount = rbm.getHiddenBiases()->size();
  const unsigned sampleCount = getVisibleVector()->size() / visibleCount;
//  std::cout << "Encode: " << sampleCount << std::endl;

  ublas::matrix<float> visibles(sampleCount, visibleCount);
  std::copy(getVisibleVector()->begin(), getVisibleVector()->end(), visibles.data().begin());
  tbblas::device_matrix<float> X(sampleCount, visibleCount);
  X = visibles;

  if (rbm.getIsGaussian()) {
    assert (rbm.getVisibleMeans() && rbm.getVisibleMeans()->size());
    assert (rbm.getVisibleStds() && rbm.getVisibleStds()->size());

    const float mean = rbm.getVisibleMeans()->data()[0];
    const float stddev = rbm.getVisibleStds()->data()[0];
    thrust::transform(X.data().begin(), X.data().end(), X.data().begin(), (_1 - mean) / stddev);
  }

  tbblas::device_matrix<float>& W = *rbm.getWeightMatrix();
  tbblas::device_vector<float>& c = *rbm.getHiddenBiases();
  tbblas::device_matrix<float> hidprobs(sampleCount, hiddenCount);

  // Calculate p(h | X, W) = sigm(XW + C)
  hidprobs = tbblas::prod(X, W);
  for (unsigned iRow = 0; iRow < hidprobs.size1(); ++iRow)
    tbblas::row(hidprobs, iRow) += c;

  switch(hiddenUnitType) {
  case HiddenUnitType::Bernoulli:
    thrust::transform(hidprobs.data().begin(), hidprobs.data().end(),
        hidprobs.data().begin(), sigmoid<float>());
    break;
  case HiddenUnitType::ReLU:
    thrust::transform(hidprobs.data().begin(), hidprobs.data().end(),
        hidprobs.data().begin(), min_0<float>());
    break;
  }

  if (getSampleHiddens() && hiddenUnitType == HiddenUnitType::Bernoulli) {
    thrust::transform(
        hidprobs.data().begin(), hidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
        hidprobs.data().begin(), sample_units<float>()
    );
  }
  ublas::matrix<float> hiddens = hidprobs;
  boost::shared_ptr<std::vector<float> > hiddenVector(new std::vector<float>(sampleCount * hiddenCount));
  std::copy(hiddens.data().begin(), hiddens.data().end(), hiddenVector->begin());
  data->setHiddenVector(hiddenVector);
}

}

}

