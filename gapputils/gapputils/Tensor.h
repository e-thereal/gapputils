/*
 * Tensor.h
 *
 *  Created on: Oct 22, 2014
 *      Author: tombr
 */

#ifndef GAPPUTILS_TENSOR_H_
#define GAPPUTILS_TENSOR_H_

#include <tbblas/tensor.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace gapputils {

typedef tbblas::tensor<float, 4> host_tensor_t;
typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

}

#endif /* GAPPUTILS_TENSOR_H_ */
