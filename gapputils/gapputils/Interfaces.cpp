/*
 * Interfaces.cpp
 *
 *  Created on: 2012-10-21
 *      Author: tombr
 */

#include "Interfaces.h"

#include <gapputils/GenerateInterfaceAttribute.h>

using namespace gapputils::attributes;

namespace gapputils {

BeginPropertyDefinitions(Interfaces)
  DefineProperty(Integer, GenerateInterface("Integer", true))
EndPropertyDefinitions

} /* namespace gapputils */
