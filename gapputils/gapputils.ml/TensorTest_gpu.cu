/*
 * TensorTest_gpu.cu
 *
 *  Created on: Mar 7, 2012
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "TensorTest.h"

#include <iostream>

#include <tbblas/device_tensor.hpp>

namespace gapputils {

namespace ml {

void TensorTest::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new TensorTest();

  typedef tbblas::device_tensor<double, 3> tensor_t;

  std::cout << "tensor test begin" << std::endl;
  {
    tensor_t tensor(4, 3, 2);
  }

  std::cout << "tensor test end" << std::endl;
}

}

}


