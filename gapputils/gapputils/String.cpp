#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class String : public gapputils::workflow::DefaultWorkflowElement<String>
{
  InitReflectableClass(String)
  
  typedef std::string property_t;
  
  Property(Value, property_t)
  
public:
  String() { setLabel("String"); }
};

BeginPropertyDefinitions(String, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<String>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class String : public gapputils::workflow::DefaultWorkflowElement<String>
{
  InitReflectableClass(String)
  
  typedef std::string property_t;
  
  Property(Value, property_t)
  
public:
  String() { setLabel("String"); }
};

BeginPropertyDefinitions(String, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<String>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
