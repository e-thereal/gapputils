/*
 * DbmModel.h
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#ifndef GML_DBMMODEL_H_
#define GML_DBMMODEL_H_

#include "Model.h"

namespace gml {

namespace convrbm4d {

struct DbmModelChecker { DbmModelChecker(); };

/** The DBM is always Gaussian-ReLU-ReLU-...-ReLU
 *
 *  In a DBM with n layers (1 visible layer and n-1 hidden layers) sizes are as follows:
 *  Weights.size() == n - 1
 *  Biases.size() == n
 *  Masks.size() == n - 1 (masks are applied the same way they are applied during layer-wise training)
 */
class DbmModel : public capputils::reflection::ReflectableClass {

public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t tensor_t;
  typedef Model::v_tensor_t v_tensor_t;
  typedef std::vector<boost::shared_ptr<v_tensor_t> > vv_tensor_t;

  friend class DbmModelChecker;

private:
  InitReflectableClass(DbmModel)

  Property(Weights, boost::shared_ptr<vv_tensor_t>)
  Property(Biases, boost::shared_ptr<v_tensor_t>)
  Property(Masks, boost::shared_ptr<v_tensor_t>)
  Property(Mean, value_t)
  Property(Stddev, value_t)

public:
  DbmModel();
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_DBMMODEL_H_ */
