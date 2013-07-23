/*
 * DbmModel_gpu.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "Model.h"

namespace gml {

namespace dbm {

ModelChecker::ModelChecker() {
  Model model;
  model.initializeClass();
  CHECK_MEMORY_LAYOUT2(Weights, model);
  CHECK_MEMORY_LAYOUT2(VisibleBias, model);
  CHECK_MEMORY_LAYOUT2(HiddenBiases, model);
  CHECK_MEMORY_LAYOUT2(Masks, model);
  CHECK_MEMORY_LAYOUT2(VisibleBlockSize, model);
  CHECK_MEMORY_LAYOUT2(Mean, model);
  CHECK_MEMORY_LAYOUT2(Stddev, model);
  CHECK_MEMORY_LAYOUT2(WeightMatrices, model);
  CHECK_MEMORY_LAYOUT2(FlatBiases, model);
}

}

}



