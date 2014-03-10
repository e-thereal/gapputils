#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class Boolean : public gapputils::workflow::DefaultWorkflowElement<Boolean>
{
  InitReflectableClass(Boolean)
  
  typedef bool property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Boolean() { setLabel("Boolean"); }
};

BeginPropertyDefinitions(Boolean, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Boolean>)
  WorkflowProperty(Description)
  WorkflowProperty(Value);
EndPropertyDefinitions

}

}
