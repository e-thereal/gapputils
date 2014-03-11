/*
 * Model.h
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#ifndef GML_DBM_MODEL_H_
#define GML_DBM_MODEL_H_

#include <capputils/reflection/ReflectableClass.h>

#include <gml.convrbm4d/Model.h>
#include <gml.rbm/Model.h>

namespace gml {

namespace dbm {

struct ModelChecker { ModelChecker(); };

/** The DBM is always Gaussian-ReLU-ReLU-...-ReLU
 *
 *  In a DBM with n layers (1 visible layer and n-1 hidden layers) sizes are as follows:
 *  Weights.size() == n - 1
 *  HiddenBiases.size() == n - 1
 *  Masks.size() == n - 1 (masks are applied the same way they are applied during layer-wise training)
 */
class Model : public capputils::reflection::ReflectableClass {

public:
  static const int dimCount = gml::convrbm4d::Model::dimCount;
  typedef gml::convrbm4d::Model::value_t value_t;
  typedef gml::convrbm4d::Model::tensor_t tensor_t;
  typedef gml::convrbm4d::Model::v_tensor_t v_tensor_t;
  typedef std::vector<boost::shared_ptr<v_tensor_t> > vv_tensor_t;
  typedef tensor_t::dim_t dim_t;

  typedef gml::rbm::Model::matrix_t matrix_t;
  typedef std::vector<boost::shared_ptr<matrix_t> > v_matrix_t;

  friend class ModelChecker;

private:
  InitReflectableClass(Model)

  Property(Weights, boost::shared_ptr<vv_tensor_t>)
  Property(VisibleBias, boost::shared_ptr<tensor_t>)
  Property(HiddenBiases, boost::shared_ptr<vv_tensor_t>)
  Property(Masks, boost::shared_ptr<v_tensor_t>)
  Property(VisibleBlockSize, dim_t)
  Property(Mean, value_t)
  Property(Stddev, value_t)

  Property(WeightMatrices, boost::shared_ptr<v_matrix_t>)
  Property(FlatBiases, boost::shared_ptr<v_matrix_t>)

public:
  Model();
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_DBM_MODEL_H_ */
