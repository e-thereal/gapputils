#include "Blurring.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Blurring)

  ReflectableBase(DefaultWorkflowElement<Blurring>)

  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(Sigma)
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

Blurring::Blurring(void) : _Sigma(1.0) {
  setLabel("Blur");
}

BlurringChecker blurringChecker;

}

}
