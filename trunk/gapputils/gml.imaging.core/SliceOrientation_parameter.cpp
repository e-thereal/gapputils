#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "SliceOrientation.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class SliceOrientation : public gapputils::workflow::DefaultWorkflowElement<SliceOrientation>
{
  InitReflectableClass(SliceOrientation)
  
  typedef gml::imaging::core::SliceOrientation property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  SliceOrientation() { setLabel("SliceOrientation"); }
};

BeginPropertyDefinitions(SliceOrientation, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<SliceOrientation>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
