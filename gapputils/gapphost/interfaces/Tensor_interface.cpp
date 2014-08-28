#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/tensor.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Tensor : public gapputils::workflow::DefaultWorkflowElement<Tensor>
{
  InitReflectableClass(Tensor)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensor() { setLabel("Tensor"); }
};

BeginPropertyDefinitions(Tensor, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensor>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Tensor : public gapputils::workflow::DefaultWorkflowElement<Tensor>
{
  InitReflectableClass(Tensor)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensor() { setLabel("Tensor"); }
};

BeginPropertyDefinitions(Tensor, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensor>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
