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

class Tensor4d : public gapputils::workflow::DefaultWorkflowElement<Tensor4d>
{
  InitReflectableClass(Tensor4d)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensor4d() { setLabel("Tensor4d"); }
};

BeginPropertyDefinitions(Tensor4d, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensor4d>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Tensor4d : public gapputils::workflow::DefaultWorkflowElement<Tensor4d>
{
  InitReflectableClass(Tensor4d)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensor4d() { setLabel("Tensor4d"); }
};

BeginPropertyDefinitions(Tensor4d, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensor4d>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
