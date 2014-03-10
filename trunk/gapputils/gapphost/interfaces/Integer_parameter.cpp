#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class Integer : public gapputils::workflow::DefaultWorkflowElement<Integer>
{
  InitReflectableClass(Integer)
  
  typedef int property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Integer() { setLabel("Integer"); }
};

BeginPropertyDefinitions(Integer, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Integer>)
  WorkflowProperty(Description)
  WorkflowProperty(Value);
EndPropertyDefinitions

}

}
