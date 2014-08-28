#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

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
  Tensors4d() { setLabel("Tensors4d"); }
};

BeginPropertyDefinitions(Tensors4d, Interface(), Renamed("interfaces::inputs::Tensors"), Deprecated("Use interfaces::inputs::Tensors instead."))
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
  Tensors4d() { setLabel("Tensors4d"); }
};

BeginPropertyDefinitions(Tensors4d, Interface(), Renamed("interfaces::outputs::Tensors"), Deprecated("Use interfaces::outputs::Tensors instead."))
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Description)
  WorkflowProperty(Values, Input("Values"), Enumerable<Type, false>())
  WorkflowProperty(Value, Input("Value"), ToEnumerable(Id - 1));
EndPropertyDefinitions

}

}
