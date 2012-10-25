/*
 * GenerateInterfaceAttribute.cpp
 *
 *  Created on: Oct 4, 2012
 *      Author: tombr
 */

#include "GenerateInterfaceAttribute.h"

namespace gapputils {

namespace attributes {

GenerateInterfaceAttribute::GenerateInterfaceAttribute(const std::string& name, const std::string& header, bool isParameter)
 : name(name), header(header), isParameter(isParameter) { }

GenerateInterfaceAttribute::~GenerateInterfaceAttribute() { }

std::string GenerateInterfaceAttribute::getName() const {
  return name;
}

std::string GenerateInterfaceAttribute::getHeader() const {
  return header;
}

bool GenerateInterfaceAttribute::getIsParameter() const {
  return isParameter;
}

capputils::attributes::AttributeWrapper* GenerateInterface(const std::string& name, const std::string& header, bool isParameter) {
  return new capputils::attributes::AttributeWrapper(new GenerateInterfaceAttribute(name, header, isParameter));
}

capputils::attributes::AttributeWrapper* GenerateInterface(const std::string& name, bool isParameter) {
  return new capputils::attributes::AttributeWrapper(new GenerateInterfaceAttribute(name, "", isParameter));
}

} /* namespace attributes */

} /* namespace gapputils */




