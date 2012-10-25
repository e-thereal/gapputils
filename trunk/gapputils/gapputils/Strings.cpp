#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class Strings : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Strings)
  
  typedef std::string property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Value, property_t)
  
public:
  Strings() : _Values(new std::vector<property_t>()) { setLabel("Strings"); }
};

BeginPropertyDefinitions(Strings, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Values, Output("Values"), Enumerable<Type, false>(), NotNull<Type>())
  WorkflowProperty(Value, Output("Value"), FromEnumerable(Id - 1));
EndPropertyDefinitions

}

namespace outputs {

class Strings : public gapputils::workflow::CollectionElement
{
  InitReflectableClass(Strings)
  
  typedef std::string property_t;
  
  Property(Values, boost::shared_ptr<std::vector<property_t> >)
  Property(Value, property_t)
  
public:
  Strings() : _Values(new std::vector<property_t>()) { setLabel("Strings"); }
};

BeginPropertyDefinitions(Strings, Interface())
  ReflectableBase(gapputils::workflow::CollectionElement)
  WorkflowProperty(Values, Input("Values"), Enumerable<Type, false>())
  WorkflowProperty(Value, Input("Value"), ToEnumerable(Id - 1));
EndPropertyDefinitions

}

}
