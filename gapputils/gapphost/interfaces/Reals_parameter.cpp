#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class Reals : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Reals)
  
  typedef double property_t;
  
  Property(Values, std::vector<property_t>)
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Reals() { setLabel("Reals"); }
};

BeginPropertyDefinitions(Reals, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Enumerable<Type, false>())
  WorkflowProperty(Value, FromEnumerable(Id - 1));
EndPropertyDefinitions

}

}
