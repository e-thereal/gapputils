/*
 * RbmModel.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmModel.h"

#include <cassert>
#include <cmath>

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/lambda/lambda.hpp>

namespace ublas = boost::numeric::ublas;
using namespace boost::lambda;

namespace gapputils {

namespace ml {

float sigmoid(const float& x) {
  return 1.f/ (1.f + exp(-x));
}

float createBernoulliSample::operator()(const float& x) const {
  return (float)(((float)rand() / (float)RAND_MAX) < x);
}

float createNormalSample::operator()(const float& x) const {
  return normal(eng) + x;
}

BeginPropertyDefinitions(RbmModel)

  DefineProperty(VisibleBiases)
  DefineProperty(HiddenBiases)
  DefineProperty(WeightMatrix)
  DefineProperty(VisibleMeans)
  DefineProperty(VisibleStds)

EndPropertyDefinitions

RbmModel::RbmModel() {
}

RbmModel::~RbmModel() {
}

boost::shared_ptr<ublas::matrix<float> > RbmModel::encodeDesignMatrix(ublas::matrix<float>& designMatrix, bool binary) const {
  boost::shared_ptr<ublas::matrix<float> > scaledMatrix(new ublas::matrix<float>(designMatrix.size1(), designMatrix.size2()));
  ublas::matrix<float>& m = *scaledMatrix;

  for (unsigned iCol = 0; iCol < m.size2(); ++iCol) {
    ublas::column(m, iCol) = (ublas::column(designMatrix, iCol) -
        ublas::scalar_vector<float>(m.size1(), (*getVisibleMeans())(iCol))) / (*getVisibleStds())(iCol);
  }

  if (binary)
    std::transform(m.data().begin(), m.data().end(), m.data().begin(), (_1 >= 0.f) * 1.f);

  return scaledMatrix;
}

boost::shared_ptr<ublas::matrix<float> > RbmModel::decodeApproximation(ublas::matrix<float>& approximation) const {
//  const unsigned visibleCount = getVisibleCount();
//
//  assert((approximation->size() % (visibleCount + 1)) == 0);
//
//  const unsigned sampleCount = approximation->size() / (visibleCount + 1);
//  std::vector<float>* visibleMeans = getVisibleMeans().get();
//  std::vector<float>* visibleStds = getVisibleStds().get();
//
//  boost::shared_ptr<std::vector<float> > scaledSet(new std::vector<float>(approximation->size() - sampleCount));
//  for (unsigned iSample = 0, iApprox = 0, iScaled = 0; iSample < sampleCount; ++iSample) {
//    ++iApprox;
//    for (unsigned iFeature = 0; iFeature < visibleCount; ++iFeature, ++iApprox, ++iScaled) {
//      scaledSet->at(iScaled) = approximation->at(iApprox) * visibleStds->at(iFeature) + visibleMeans->at(iFeature);
//    }
//  }
//
//  return scaledSet;
}

}

}
