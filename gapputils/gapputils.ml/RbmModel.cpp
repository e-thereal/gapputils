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
#include <boost/numeric/ublas/io.hpp>
#include <boost/lambda/lambda.hpp>

#include <capputils/SerializeAttribute.h>

#include <iostream>

#include "tbblas_serialize.hpp"

namespace ublas = boost::numeric::ublas;
using namespace boost::lambda;

using namespace capputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmModel)

  DefineProperty(VisibleBiases, Serialize<TYPE_OF(VisibleBiases)>())
  DefineProperty(HiddenBiases, Serialize<TYPE_OF(HiddenBiases)>())
  DefineProperty(WeightMatrix, Serialize<TYPE_OF(WeightMatrix)>())
  DefineProperty(VisibleMeans, Serialize<TYPE_OF(VisibleMeans)>())
  DefineProperty(VisibleStds, Serialize<TYPE_OF(VisibleStds)>())

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
//  std::cout << "Design: " << designMatrix << std::endl;
//  std::cout << "Means: " << *getVisibleMeans() << std::endl;
//  std::cout << "Stddev: " << *getVisibleStds() << std::endl;
//  std::cout << "Scaled: " << m << std::endl;

  if (binary)
    std::transform(m.data().begin(), m.data().end(), m.data().begin(), (_1 > 0.f) * 1.f);

  return scaledMatrix;
}

boost::shared_ptr<ublas::matrix<float> > RbmModel::decodeApproximation(ublas::matrix<float>& approximation) const {
  boost::shared_ptr<ublas::matrix<float> > unscaledMatrix(new ublas::matrix<float>(approximation.size1(), approximation.size2()));
  ublas::matrix<float>& m = *unscaledMatrix;

  for (unsigned iCol = 0; iCol < m.size2(); ++iCol) {
    ublas::column(m, iCol) = ublas::column(approximation, iCol) * (*getVisibleStds())(iCol) +
        ublas::scalar_vector<float>(m.size1(), (*getVisibleMeans())(iCol));
  }

  return unscaledMatrix;
}

}

}
