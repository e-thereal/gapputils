/*
 * Model.h
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#ifndef GML_MODEL_H_
#define GML_MODEL_H_

#include <capputils/reflection/ReflectableClass.h>

#include "DimensionalityReductionMethod.h"

namespace yala {

template<class T> class YADimReduce;

}

namespace gml {

namespace dimreduce {

class Model : public capputils::reflection::ReflectableClass {
public:
  typedef double value_t;

  InitReflectableClass(Model)

  Property(Model, boost::shared_ptr<yala::YADimReduce<value_t> >)
  Property(Method, DimensionalityReductionMethod)

public:
  Model();
};

} /* namespace dimreduce */
} /* namespace gml */
#endif /* MODEL_H_ */
