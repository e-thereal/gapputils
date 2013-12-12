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
  Model model;
  model.initializeClass();
  CHECK_MEMORY_LAYOUT2(Filters, model);
  CHECK_MEMORY_LAYOUT2(VisibleBias, model);
  CHECK_MEMORY_LAYOUT2(HiddenBiases, model);
  CHECK_MEMORY_LAYOUT2(FilterKernelSize, model);
  CHECK_MEMORY_LAYOUT2(Mean, model);
  CHECK_MEMORY_LAYOUT2(Stddev, model);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, model);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, model);
  CHECK_MEMORY_LAYOUT2(Mask, model);
  CHECK_MEMORY_LAYOUT2(ConvolutionType, model);
}

}

}
