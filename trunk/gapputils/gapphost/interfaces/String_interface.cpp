#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class String : public gapputils::workflow::DefaultWorkflowElement<String>
{
  InitReflectableClass(String)
  
  typedef std::string property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  String() { setLabel("String"); }
};

BeginPropertyDefinitions(String, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<String>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class String : public gapputils::workflow::DefaultWorkflowElement<String>
{
  InitReflectableClass(String)
  
  typedef std::string property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  String() { setLabel("String"); }
};

BeginPropertyDefinitions(String, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<String>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
