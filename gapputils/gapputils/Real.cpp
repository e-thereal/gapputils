#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Real : public gapputils::workflow::DefaultWorkflowElement<Real>
{
  InitReflectableClass(Real)
  
  typedef double property_t;
  
  Property(Value, property_t)
  
public:
  Real() { setLabel("Real"); }
};

BeginPropertyDefinitions(Real, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Real>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Real : public gapputils::workflow::DefaultWorkflowElement<Real>
{
  InitReflectableClass(Real)
  
  typedef double property_t;
  
  Property(Value, property_t)
  
public:
  Real() { setLabel("Real"); }
};

BeginPropertyDefinitions(Real, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Real>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
