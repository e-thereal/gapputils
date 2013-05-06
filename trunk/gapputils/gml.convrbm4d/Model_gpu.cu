/*
 * Model_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Model.h"

namespace gml {

namespace convrbm4d {

ModelChecker::ModelChecker() {
  Model test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Filters, test);
  CHECK_MEMORY_LAYOUT2(VisibleBias, test);
  CHECK_MEMORY_LAYOUT2(HiddenBiases, test);
  CHECK_MEMORY_LAYOUT2(FilterKernelSize, test);
  CHECK_MEMORY_LAYOUT2(Mean, test);
  CHECK_MEMORY_LAYOUT2(Stddev, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
}

}

}
