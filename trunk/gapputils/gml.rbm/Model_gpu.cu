/*
 * Model_gpu.cu
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#include "Model.h"

namespace gml {

namespace rbm {

ModelChecker::ModelChecker() {
  Model test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(VisibleBiases, test);
  CHECK_MEMORY_LAYOUT2(HiddenBiases, test);
  CHECK_MEMORY_LAYOUT2(WeightMatrix, test);
  CHECK_MEMORY_LAYOUT2(Mean, test);
  CHECK_MEMORY_LAYOUT2(Stddev, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
}

}

}
