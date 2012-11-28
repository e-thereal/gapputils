/*
 * Model.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <capputils/ReflectableClass.h>

#include <tbblas/tensor.hpp>

#include "UnitType.h"

namespace gml {

namespace convrbm {

struct ModelChecker { ModelChecker(); };

class Model : public capputils::reflection::ReflectableClass {
public:
  const static unsigned dimCount = 3;
  typedef double value_t;
  typedef tbblas::tensor<value_t, dimCount, false> tensor_t;

  friend class ModelChecker;

private:
  InitReflectableClass(Model)

  int dummy; ///< needed to align GCC and NVCC memory layouts.

  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(VisibleBias, value_t)
  Property(HiddenBiases, boost::shared_ptr<std::vector<value_t> >)
  Property(Mean, value_t)
  Property(Stddev, value_t)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)

public:
  Model();
  virtual ~Model();

  boost::shared_ptr<Model> clone();
};

}

} /* namespace gml */

#endif /* MODEL_H_ */
