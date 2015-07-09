#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/encoder_model.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace inputs {

class EncoderModel : public gapputils::workflow::DefaultWorkflowElement<EncoderModel>
{
  InitReflectableClass(EncoderModel)
  
  typedef boost::shared_ptr<tbblas::deeplearn::encoder_model<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  EncoderModel() { setLabel("EncoderModel"); }
};

BeginPropertyDefinitions(EncoderModel, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<EncoderModel>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Output(""));
EndPropertyDefinitions

}

namespace outputs {

class EncoderModel : public gapputils::workflow::DefaultWorkflowElement<EncoderModel>
{
  InitReflectableClass(EncoderModel)
  
  typedef boost::shared_ptr<tbblas::deeplearn::encoder_model<float,4> > property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  EncoderModel() { setLabel("EncoderModel"); }
};

BeginPropertyDefinitions(EncoderModel, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<EncoderModel>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Input(""));
EndPropertyDefinitions

}

}
