/*
 * RbmModel.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmModel.h"

#include <cassert>

#include <cmath>

namespace gapputils {

namespace ml {

float sigmoid(const float& x) {
  return 1.f/ (1.f + exp(-x));
}

BeginPropertyDefinitions(RbmModel)

  DefineProperty(VisibleCount)
  DefineProperty(HiddenCount)
  DefineProperty(WeightMatrix)
  DefineProperty(VisibleMeans)
  DefineProperty(VisibleStds)

EndPropertyDefinitions

RbmModel::RbmModel() : _VisibleCount(1), _HiddenCount(1) {
}

RbmModel::~RbmModel() {
}

boost::shared_ptr<std::vector<float> > RbmModel::encodeDesignMatrix(std::vector<float>* designMatrix) const {
  const unsigned visibleCount = getVisibleCount();

  assert((designMatrix->size() % visibleCount) == 0);

  const unsigned sampleCount = designMatrix->size() / visibleCount;
  std::vector<float>* visibleMeans = getVisibleMeans().get();
  std::vector<float>* visibleStds = getVisibleStds().get();

  boost::shared_ptr<std::vector<float> > scaledSet(new std::vector<float>(designMatrix->size() + sampleCount));
  for (unsigned iSample = 0, iTraining = 0, iScaled = 0; iSample < sampleCount; ++iSample) {
    scaledSet->at(iScaled++) = 1.f;
    for (unsigned iFeature = 0; iFeature < visibleCount; ++iFeature, ++iTraining, ++iScaled) {
      scaledSet->at(iScaled) = (designMatrix->at(iTraining) - visibleMeans->at(iFeature)) / visibleStds->at(iFeature);
    }
  }

  return scaledSet;
}

boost::shared_ptr<std::vector<float> > RbmModel::decodeApproximation(std::vector<float>* approximation) const {
  const unsigned visibleCount = getVisibleCount();

  assert((approximation->size() % (visibleCount + 1)) == 0);

  const unsigned sampleCount = approximation->size() / (visibleCount + 1);
  std::vector<float>* visibleMeans = getVisibleMeans().get();
  std::vector<float>* visibleStds = getVisibleStds().get();

  boost::shared_ptr<std::vector<float> > scaledSet(new std::vector<float>(approximation->size() - sampleCount));
  for (unsigned iSample = 0, iApprox = 0, iScaled = 0; iSample < sampleCount; ++iSample) {
    ++iApprox;
    for (unsigned iFeature = 0; iFeature < visibleCount; ++iFeature, ++iApprox, ++iScaled) {
      scaledSet->at(iScaled) = approximation->at(iApprox) * visibleStds->at(iFeature) + visibleMeans->at(iFeature);
    }
  }

  return scaledSet;
}

}

}
