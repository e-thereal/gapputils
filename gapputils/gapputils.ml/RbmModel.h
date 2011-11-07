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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <random>

namespace gapputils {

namespace ml {

/**
 * \brief The logistic function or sigmoid function
 */
float sigmoid(const float& x);

/**
 * \brief Samples from a Bernoulli distribution given the mean
 */
struct createBernoulliSample {
  float operator()(const float& x) const;
};

/**
 * \brief Sampling from a normal distribution with mean \c x and variance 1
 */
struct createNormalSample {
  mutable std::ranlux64_base_01 eng;
  mutable std::normal_distribution<float> normal;

  float operator()(const float& x) const;
};

/**
 * \brief Contains bias terms and weight matrix of an RBM plus statistics for feature scaling
 */
class RbmModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(RbmModel)

  Property(VisibleBiases, boost::shared_ptr<boost::numeric::ublas::vector<float> >)
  Property(HiddenBiases, boost::shared_ptr<boost::numeric::ublas::vector<float> >)
  Property(WeightMatrix, boost::shared_ptr<boost::numeric::ublas::matrix<float> >)
  Property(VisibleMeans, boost::shared_ptr<boost::numeric::ublas::vector<float> >)  ///< used for feature scaling
  Property(VisibleStds, boost::shared_ptr<boost::numeric::ublas::vector<float> >)   ///< used for feature scaling

public:
  RbmModel();
  virtual ~RbmModel();

  /**
   * \brief Normalizes the data using the RBM statistics
   */
  boost::shared_ptr<boost::numeric::ublas::matrix<float> > encodeDesignMatrix(boost::numeric::ublas::matrix<float>& designMatrix, bool binary) const;

  /**
   * \brief Scales the approximation using the RBM statistics
   */
  boost::shared_ptr<boost::numeric::ublas::matrix<float> > decodeApproximation(boost::numeric::ublas::matrix<float>& approximation) const;
};

}

}


#endif /* GAPPUTILS_ML_RBMMODEL_H_ */
