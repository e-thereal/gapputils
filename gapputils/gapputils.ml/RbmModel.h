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
#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

#include <cuda_runtime.h>

namespace gapputils {

namespace ml {

/**
 * \brief The logistic function or sigmoid function
 */
template<class T>
struct sigmoid {

__host__ __device__ T operator()(const T& x) const {
  return 1.f/ (1.f + exp(-x));
}

};

/**
 * \brief Contains bias terms and weight matrix of an RBM plus statistics for feature scaling
 */
class RbmModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(RbmModel)

  Property(VisibleBiases, boost::shared_ptr<tbblas::device_vector<float> >)
  Property(HiddenBiases, boost::shared_ptr<tbblas::device_vector<float> >)
  Property(WeightMatrix, boost::shared_ptr<tbblas::device_matrix<float> >)
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
