#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/activation_function.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class ActivationFunction : public gapputils::workflow::DefaultWorkflowElement<ActivationFunction>
{
  InitReflectableClass(ActivationFunction)
  
  typedef tbblas::deeplearn::activation_function property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  ActivationFunction() { setLabel("ActivationFunction"); }
};

BeginPropertyDefinitions(ActivationFunction, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<ActivationFunction>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
