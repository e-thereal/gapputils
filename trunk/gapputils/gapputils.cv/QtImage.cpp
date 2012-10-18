#include <gapputils/DefaultWorkflowElement.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include <qimage.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class QtImage : public gapputils::workflow::DefaultWorkflowElement<QtImage>
{
  InitReflectableClass(QtImage)
  
  typedef boost::shared_ptr<QImage> property_t;
  
  Property(Value, property_t)
  
public:
  QtImage() { setLabel("QtImage"); }
};

BeginPropertyDefinitions(QtImage, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<QtImage>)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class QtImage : public gapputils::workflow::DefaultWorkflowElement<QtImage>
{
  InitReflectableClass(QtImage)
  
  typedef boost::shared_ptr<QImage> property_t;
  
  Property(Value, property_t)
  
public:
  QtImage() { setLabel("QtImage"); }
};

BeginPropertyDefinitions(QtImage, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<QtImage>)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
