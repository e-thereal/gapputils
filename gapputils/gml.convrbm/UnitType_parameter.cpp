#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "UnitType.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class UnitType : public gapputils::workflow::DefaultWorkflowElement<UnitType>
{
  InitReflectableClass(UnitType)
  
  typedef gml::convrbm::UnitType property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  UnitType() { setLabel("UnitType"); }
};

BeginPropertyDefinitions(UnitType, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<UnitType>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
