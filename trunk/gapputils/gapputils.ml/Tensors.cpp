#include <gapputils/DefaultWorkflowElement.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include <tbblas/tensor.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Tensors : public gapputils::workflow::DefaultWorkflowElement<Tensors>
{
  InitReflectableClass(Tensors)
  
  typedef boost::shared_ptr<std::vector<boost::shared_ptr<tbblas::tensor<double, 3u, false> >, std::allocator<boost::shared_ptr<tbblas::tensor<double, 3u, false> > > > > property_t;
  
  Property(Value, property_t)
  
public:
  Tensors() { setLabel("Tensors"); }
};

BeginPropertyDefinitions(Tensors, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensors>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class Tensors : public gapputils::workflow::DefaultWorkflowElement<Tensors>
{
  InitReflectableClass(Tensors)
  
  typedef boost::shared_ptr<std::vector<boost::shared_ptr<tbblas::tensor<double, 3u, false> >, std::allocator<boost::shared_ptr<tbblas::tensor<double, 3u, false> > > > > property_t;
  
  Property(Value, property_t)
  
public:
  Tensors() { setLabel("Tensors"); }
};

BeginPropertyDefinitions(Tensors, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Tensors>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
