/*
 * DbmModel.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "DbmModel.h"

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(DbmModel)
  DefineProperty(Weights, Serialize<Type>())
  DefineProperty(VisibleBias, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(Masks, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
EndPropertyDefinitions

DbmModel::DbmModel() : _Mean(0.0), _Stddev(1.0) { }

DbmModelChecker dbmModelChecker;

} /* namespace convrbm4d */

} /* namespace gml */
