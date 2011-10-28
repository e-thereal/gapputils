/*
 * RbmModel.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMMODEL_H_
#define GAPPUTILS_ML_RBMMODEL_H_

#include <capputils/ReflectableClass.h>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace ml {

/**
 * \brief The logistic function or sigmoid function
 */
float sigmoid(const float& x);

/**
 * \brief Contains bias terms and weight matrix of an RBM plus statistics for feature scaling
 *
 * Bias terms are absorbed into the weight matrix
 *
 * W' = / 0 cT \
 *      \ b W  /
 *
 * with b = visible bias, c = hidden bias and W = VxH weight matrix.
 *
 * Thus:
 *
 * E = -(xTWh + xTb + hTc) = -x'TW'h'
 *
 * with x' = (1 xT)T and h' = (1 hT)T
 */
class RbmModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(RbmModel)

  Property(VisibleCount, int)
  Property(HiddenCount, int)
  Property(WeightMatrix, boost::shared_ptr<std::vector<float> >)
  Property(VisibleMeans, boost::shared_ptr<std::vector<float> >)  ///< used for feature scaling
  Property(VisibleStds, boost::shared_ptr<std::vector<float> >)   ///< used for feature scaling

public:
  RbmModel();
  virtual ~RbmModel();

  /**
   * \brief Normalizes the data using the RBM statistics and adds ones before the first column vector
   */
  boost::shared_ptr<std::vector<float> > encodeDesignMatrix(std::vector<float>* designMatrix) const;

  /**
   * \brief Scales the approximation using the RBM statistics and crops the first column vector
   */
  boost::shared_ptr<std::vector<float> > decodeApproximation(std::vector<float>* approximation) const;
};

}

}


#endif /* GAPPUTILS_ML_RBMMODEL_H_ */
