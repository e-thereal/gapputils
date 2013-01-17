#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "UnitType.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class RbmUnitType : public gapputils::workflow::DefaultWorkflowElement<RbmUnitType>
{
  InitReflectableClass(RbmUnitType)
  
  typedef gml::rbm::UnitType property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  RbmUnitType() { setLabel("RbmUnitType"); }
};

BeginPropertyDefinitions(RbmUnitType, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<RbmUnitType>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}