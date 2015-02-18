#include "Blurring.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Blurring)

  ReflectableBase(DefaultWorkflowElement<Blurring>)

  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(SigmaX)
  WorkflowProperty(SigmaY)
  WorkflowProperty(SigmaZ)
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

Blurring::Blurring(void) : _SigmaX(1.0),_SigmaY(1.0), _SigmaZ(1.0) {
  setLabel("Blur");
}

BlurringChecker blurringChecker;

}

}
