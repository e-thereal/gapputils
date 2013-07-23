/*
 * Model.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "Model.h"

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;

namespace gml {

namespace dbm {

BeginPropertyDefinitions(Model)
  DefineProperty(Weights, Serialize<Type>())
  DefineProperty(VisibleBias, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(Masks, Serialize<Type>())
  DefineProperty(VisibleBlockSize, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
  DefineProperty(WeightMatrices, Serialize<Type>())
  DefineProperty(FlatBiases, Serialize<Type>())
EndPropertyDefinitions

Model::Model() : _Mean(0.0), _Stddev(1.0) { }

ModelChecker modelChecker;

} /* namespace dbm */

} /* namespace gml */
