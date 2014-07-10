#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

#include "tbblas/deeplearn/convolution_type.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {
  
namespace parameters {

class ConvolutionType : public gapputils::workflow::DefaultWorkflowElement<ConvolutionType>
{
  InitReflectableClass(ConvolutionType)
  
  typedef tbblas::deeplearn::convolution_type property_t;
  
  Property(Description, std::string)
  Property(Value, property_t)
  
public:
  ConvolutionType() { setLabel("ConvolutionType"); }
};

BeginPropertyDefinitions(ConvolutionType, Interface())
  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<ConvolutionType>)
  WorkflowProperty(Description)
  WorkflowProperty(Value, Enumerator<Type>());
EndPropertyDefinitions

}

}
