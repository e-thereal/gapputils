/*
 * InterfaceAttribute.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "InterfaceAttribute.h"

namespace gapputils {

namespace attributes {

InterfaceAttribute::InterfaceAttribute() {
}

InterfaceAttribute::~InterfaceAttribute() {
}

capputils::attributes::AttributeWrapper* Interface() {
  return new capputils::attributes::AttributeWrapper(new InterfaceAttribute());
}

} /* namespace attributes */

} /* namespace gapputils */
