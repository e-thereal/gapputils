#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class RealVector : public gapputils::workflow::DefaultWorkflowElement<RealVector>
{
  InitReflectableClass(RealVector)
  
  typedef boost::shared_ptr<std::vector<double> > property_t;
  
  Property(Value, property_t)
  
public:
  RealVector() { setLabel("RealVector"); }
};

BeginPropertyDefinitions(RealVector, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<RealVector>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class RealVector : public gapputils::workflow::DefaultWorkflowElement<RealVector>
{
  InitReflectableClass(RealVector)
  
  typedef boost::shared_ptr<std::vector<double> > property_t;
  
  Property(Value, property_t)
  
public:
  RealVector() { setLabel("RealVector"); }
};

BeginPropertyDefinitions(RealVector, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<RealVector>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
