#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class Boolean : public gapputils::workflow::DefaultWorkflowElement<Boolean>
{
  InitReflectableClass(Boolean)
  
  typedef bool property_t;
  
  Property(Value, property_t)
  
public:
  Boolean() { setLabel("Boolean"); }
};

BeginPropertyDefinitions(Boolean, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Boolean>)
  WorkflowProperty(Value);
EndPropertyDefinitions

}

}
