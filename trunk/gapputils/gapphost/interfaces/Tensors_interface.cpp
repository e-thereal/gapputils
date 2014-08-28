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

class Tensors : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Tensors)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensors() { setLabel("Tensors"); }
};

BeginPropertyDefinitions(Tensors, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Output("Values"), Enumerable<Type, false>(), NotNull<Type>())
  WorkflowProperty(Value, Output("Value"), FromEnumerable(Id - 1));
EndPropertyDefinitions

}

namespace outputs {

class Tensors : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Tensors)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensors() { setLabel("Tensors"); }
};

BeginPropertyDefinitions(Tensors, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Input("Values"), Enumerable<Type, false>())
  WorkflowProperty(Value, Input("Value"), ToEnumerable(Id - 1));
EndPropertyDefinitions

}

}
