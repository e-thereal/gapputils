#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/sparsity_method.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class SparsityMethod : public gapputils::workflow::DefaultWorkflowElement<SparsityMethod>
{
  InitReflectableClass(SparsityMethod)
  
  typedef tbblas::deeplearn::sparsity_method property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  SparsityMethod() { setLabel("SparsityMethod"); }
};

BeginPropertyDefinitions(SparsityMethod, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<SparsityMethod>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
