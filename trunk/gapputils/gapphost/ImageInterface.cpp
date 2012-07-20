#include "ImageInterface.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/ReadOnlyAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace inputs {

BeginPropertyDefinitions(Image, Interface())
  DefineProperty(Value, Output(""), ReadOnly(), Volatile(), Observe(PROPERTY_ID))
EndPropertyDefinitions

Image::Image(void) {
  setLabel("Image");
}

}

namespace outputs {

BeginPropertyDefinitions(Image, Interface())
  DefineProperty(Value, Input(""), ReadOnly(), Volatile(), Observe(PROPERTY_ID))
EndPropertyDefinitions

Image::Image(void) {
  setLabel("Image");
}

}

}
