/*
 * GroupAttribute.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: tombr
 */

#include "GroupAttribute.h"

namespace gapputils {

namespace attributes {

GroupAttribute::GroupAttribute(const std::string& name) : name(name) { }

GroupAttribute::~GroupAttribute() { }

const std::string& GroupAttribute::getName() const {
  return name;
}

capputils::attributes::AttributeWrapper* Group(const std::string& name) {
  return new capputils::attributes::AttributeWrapper(new GroupAttribute(name));
}

}

}
