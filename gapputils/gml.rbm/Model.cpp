/*
 * RbmModel.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "Model.h"

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;

namespace gml {

namespace rbm {

BeginPropertyDefinitions(Model)

  DefineProperty(VisibleBiases, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(WeightMatrix, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
  DefineProperty(VisibleUnitType, Serialize<Type>())
  DefineProperty(HiddenUnitType, Serialize<Type>())

EndPropertyDefinitions

Model::Model() { }

ModelChecker modelChecker;

}

}
