#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "DropoutMethod.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class DropoutMethod : public gapputils::workflow::DefaultWorkflowElement<DropoutMethod>
{
  InitReflectableClass(DropoutMethod)
  
  typedef gml::convrbm4d::DropoutMethod property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  DropoutMethod() { setLabel("DropoutMethod"); }
};

BeginPropertyDefinitions(DropoutMethod, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<DropoutMethod>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
