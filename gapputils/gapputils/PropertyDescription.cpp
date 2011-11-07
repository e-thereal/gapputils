/*
 * PropertyDescription.cpp
 *
 *  Created on: Aug 16, 2011
 *      Author: tombr
 */

#include "PropertyDescription.h"

#include <capputils/EnumerableAttribute.h>

namespace gapputils {

BeginPropertyDefinitions(PropertyDescription)
  using namespace capputils::attributes;

  DefineProperty(Name)
  DefineProperty(Type)
  DefineProperty(DefaultValue)
  DefineProperty(PropertyAttributes, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>())
EndPropertyDefinitions

PropertyDescription::PropertyDescription()
  : _PropertyAttributes(new std::vector<std::string>())
{
}

PropertyDescription::~PropertyDescription() {
}

} /* namespace gapputils */