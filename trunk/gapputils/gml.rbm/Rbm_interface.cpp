#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "Model.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Rbm : public gapputils::workflow::DefaultWorkflowElement<Rbm>
{
  InitReflectableClass(Rbm)
  
  typedef boost::shared_ptr<gml::rbm::Model> property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Rbm() { setLabel("Rbm"); }
};

BeginPropertyDefinitions(Rbm, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Rbm>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Rbm : public gapputils::workflow::DefaultWorkflowElement<Rbm>
{
  InitReflectableClass(Rbm)
  
  typedef boost::shared_ptr<gml::rbm::Model> property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Rbm() { setLabel("Rbm"); }
};

BeginPropertyDefinitions(Rbm, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Rbm>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
