#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include "tbblas/tensor.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Tensors4d : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Tensors4d)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensors4d() : _Values(new std::vector<property_t>()) { setLabel("Tensors4d"); }
};

BeginPropertyDefinitions(Tensors4d, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Output("Values"), Enumerable<Type, false>(), NotNull<Type>())
  WorkflowProperty(Value, Output("Value"), FromEnumerable(Id - 1));
EndPropertyDefinitions

}

namespace outputs {

class Tensors4d : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Tensors4d)
  
  typedef boost::shared_ptr<tbblas::tensor<float,4> > property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  Tensors4d() : _Values(new std::vector<property_t>()) { setLabel("Tensors4d"); }
};

BeginPropertyDefinitions(Tensors4d, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Input("Values"), Enumerable<Type, false>())
  WorkflowProperty(Value, Input("Value"), ToEnumerable(Id - 1));
EndPropertyDefinitions

}

}
