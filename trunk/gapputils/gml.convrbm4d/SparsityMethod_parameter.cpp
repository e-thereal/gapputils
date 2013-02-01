#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "SparsityMethod.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class SparsityMethod : public gapputils::workflow::DefaultWorkflowElement<SparsityMethod>
{
  InitReflectableClass(SparsityMethod)
  
  typedef gml::convrbm4d::SparsityMethod property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  SparsityMethod() { setLabel("SparsityMethod"); }
};

BeginPropertyDefinitions(SparsityMethod, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<SparsityMethod>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
