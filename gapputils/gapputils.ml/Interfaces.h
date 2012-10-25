/*
 * Interfaces.h
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_INTERFACES_H_
#define GAPPUTILS_ML_INTERFACES_H_

#include <capputils/ReflectableClass.h>

#include <tbblas/tensor.hpp>

namespace gapputils {
namespace ml {

class Interfaces : public capputils::reflection::ReflectableClass {

  typedef tbblas::tensor<double, 3, false> host_tensor_t;

  InitReflectableClass(Interfaces)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_INTERFACES_H_ */
