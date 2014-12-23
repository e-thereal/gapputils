#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/objective_function.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class ObjectiveFunction : public gapputils::workflow::DefaultWorkflowElement<ObjectiveFunction>
{
  InitReflectableClass(ObjectiveFunction)
  
  typedef tbblas::deeplearn::objective_function property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  ObjectiveFunction() { setLabel("ObjectiveFunction"); }
};

BeginPropertyDefinitions(ObjectiveFunction, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<ObjectiveFunction>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
