/*
 * Model.cpp
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#include "Model.h"

#include <capputils/SerializeAttribute.h>

namespace gml {

namespace dimreduce {

BeginPropertyDefinitions(Model)
  using namespace capputils::attributes;

  DefineProperty(Model)
  DefineProperty(Method, Serialize<Type>())

EndPropertyDefinitions

Model::Model() {
}

} /* namespace dimreduce */
} /* namespace gml */
