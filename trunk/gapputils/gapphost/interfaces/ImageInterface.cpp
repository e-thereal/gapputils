#include "ImageInterface.h"

#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>

#include <gapputils/attributes/ReadOnlyAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace inputs {

BeginPropertyDefinitions(Image, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Image>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

Image::Image(void) {
  setLabel("Image");
}

}

namespace outputs {

BeginPropertyDefinitions(Image, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Image>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

Image::Image(void) {
  setLabel("Image");
}

}

}
