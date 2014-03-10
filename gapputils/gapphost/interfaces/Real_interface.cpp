#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Real : public gapputils::workflow::DefaultWorkflowElement<Real>
{
  InitReflectableClass(Real)
  
  typedef double property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Real() { setLabel("Real"); }
};

BeginPropertyDefinitions(Real, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Real>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Real : public gapputils::workflow::DefaultWorkflowElement<Real>
{
  InitReflectableClass(Real)
  
  typedef double property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Real() { setLabel("Real"); }
};

BeginPropertyDefinitions(Real, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Real>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
