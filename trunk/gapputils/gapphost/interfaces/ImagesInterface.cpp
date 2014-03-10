/*
 * ImagesInterface.cpp
 *
 *  Created on: Jul 26, 2012
 *      Author: tombr
 */

#include "ImagesInterface.h"

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace inputs {

BeginPropertyDefinitions(Images, Interface())
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Values, Output("Images"), Enumerable<Type, false>(), NotNull<Type>());
  WorkflowProperty(Value, Output("Image"), FromEnumerable(Id - 1));

EndPropertyDefinitions

Images::Images() {
  setLabel("Images");
}

}

namespace outputs {

BeginPropertyDefinitions(Images, Interface())
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Values, Input("Images"), Enumerable<Type, false>());
  WorkflowProperty(Value, Input("Image"), ToEnumerable(Id - 1));

EndPropertyDefinitions

Images::Images() {
  setLabel("Images");
}

}

} /* namespace interfaces */
