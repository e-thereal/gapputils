#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/conv_rbm_model.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Crbm4d : public gapputils::workflow::DefaultWorkflowElement<Crbm4d>
{
  InitReflectableClass(Crbm4d)
  
  typedef boost::shared_ptr<tbblas::deeplearn::conv_rbm_model<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Crbm4d() { setLabel("Crbm4d"); }
};

BeginPropertyDefinitions(Crbm4d, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Crbm4d>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Crbm4d : public gapputils::workflow::DefaultWorkflowElement<Crbm4d>
{
  InitReflectableClass(Crbm4d)
  
  typedef boost::shared_ptr<tbblas::deeplearn::conv_rbm_model<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Crbm4d() { setLabel("Crbm4d"); }
};

BeginPropertyDefinitions(Crbm4d, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Crbm4d>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
