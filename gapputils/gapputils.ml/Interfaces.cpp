/*
 * Interfaces.cpp
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#include "Interfaces.h"

#include <gapputils/GenerateInterfaceAttribute.h>

using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(Interfaces)

  DefineProperty(Tensors, GenerateInterface("Tensors", "tbblas/tensor.hpp"))

EndPropertyDefinitions

} /* namespace ml */
} /* namespace gapputils */
