#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/pooling_method.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class PoolingMethod : public gapputils::workflow::DefaultWorkflowElement<PoolingMethod>
{
  InitReflectableClass(PoolingMethod)
  
  typedef tbblas::deeplearn::pooling_method property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  PoolingMethod() { setLabel("PoolingMethod"); }
};

BeginPropertyDefinitions(PoolingMethod, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<PoolingMethod>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
