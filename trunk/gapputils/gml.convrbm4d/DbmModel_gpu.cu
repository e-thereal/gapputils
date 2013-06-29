/*
 * DbmModel_gpu.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "DbmModel.h"

namespace gml {

namespace convrbm4d {

DbmModelChecker::DbmModelChecker() {
  DbmModel model;
  model.initializeClass();
  CHECK_MEMORY_LAYOUT2(Weights, model);
  CHECK_MEMORY_LAYOUT2(Biases, model);
  CHECK_MEMORY_LAYOUT2(Masks, model);
  CHECK_MEMORY_LAYOUT2(Mean, model);
  CHECK_MEMORY_LAYOUT2(Stddev, model);
}

}

}



