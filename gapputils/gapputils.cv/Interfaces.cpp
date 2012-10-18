/*
 * Interfaces.cpp
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#include "Interfaces.h"

#include <gapputils/GenerateInterfaceAttribute.h>

using namespace gapputils::attributes;

namespace gapputils {
namespace cv {

BeginPropertyDefinitions(Interfaces)

  DefineProperty(Image, GenerateInterface("QtImage", "qimage.h"))

EndPropertyDefinitions

} /* namespace ml */
} /* namespace gapputils */
