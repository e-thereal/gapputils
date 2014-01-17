/*
 * Model.h
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#ifndef GML_RBM_MODEL_H_
#define GML_RBM_MODEL_H_

#include <capputils/ReflectableClass.h>

#include <tbblas/tensor.hpp>

#include "UnitType.h"

namespace gml {

namespace rbm {

struct ModelChecker { ModelChecker(); };

/**
 * \brief Contains bias terms and weight matrix of an RBM plus statistics for feature scaling
 *
 * The visible mask is used to mask out visible units that are not needed. Useful when receiving
 * the hidden units of a masked convRBM as input.
 */
class Model : public capputils::reflection::ReflectableClass {
public:
  typedef double value_t;
  typedef tbblas::tensor<value_t, 2> matrix_t;
  typedef matrix_t::dim_t dim_t;

  friend class ModelChecker;

  InitReflectableClass(Model)

  Property(VisibleBiases, boost::shared_ptr<matrix_t>)
  Property(HiddenBiases, boost::shared_ptr<matrix_t>)
  Property(WeightMatrix, boost::shared_ptr<matrix_t>)
  Property(Mean, boost::shared_ptr<matrix_t>)         ///< A 1 x visibleCount matrix
  Property(Stddev, boost::shared_ptr<matrix_t>)       ///< A 1 x visibleCount matrix
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)
  Property(VisibleMask, boost::shared_ptr<matrix_t>)  ///< A 1 x visibleCount matrix

public:
  Model();
};

}

}

#endif /* GML_RBM_MODEL_H_ */
