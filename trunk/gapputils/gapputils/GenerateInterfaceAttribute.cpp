/*
 * GenerateInterfaceAttribute.cpp
 *
 *  Created on: Oct 4, 2012
 *      Author: tombr
 */

#include "GenerateInterfaceAttribute.h"

namespace gapputils {

namespace attributes {

GenerateInterfaceAttribute::GenerateInterfaceAttribute(const std::string& name, const std::string& header)
 : name(name), header(header) { }

GenerateInterfaceAttribute::~GenerateInterfaceAttribute() { }

std::string GenerateInterfaceAttribute::getName() const {
  return name;
}

std::string GenerateInterfaceAttribute::getHeader() const {
  return header;
}

capputils::attributes::AttributeWrapper* GenerateInterface(const std::string& name, const std::string& header) {
  return new capputils::attributes::AttributeWrapper(new GenerateInterfaceAttribute(name, header));
}

} /* namespace attributes */

} /* namespace gapputils */




