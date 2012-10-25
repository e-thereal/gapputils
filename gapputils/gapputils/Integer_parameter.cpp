#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class Integer : public gapputils::workflow::DefaultWorkflowElement<Integer>
{
  InitReflectableClass(Integer)
  
  typedef int property_t;
  
  Property(Value, property_t)
  
public:
  Integer() { setLabel("Integer"); }
};

BeginPropertyDefinitions(Integer, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Integer>)
  WorkflowProperty(Value);
EndPropertyDefinitions

}

}
