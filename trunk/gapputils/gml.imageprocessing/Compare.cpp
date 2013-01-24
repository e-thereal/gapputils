/*
 * Compare.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: tombr
 */

#include "Compare.h"

#include <capputils/EventHandler.h>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(MeasureParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(NoMeasureParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(SsimParameters)

  DefineProperty(WindowWidth, Observe(Id))
  DefineProperty(WindowHeight, Observe(Id))
  DefineProperty(WindowDepth, Observe(Id))

EndPropertyDefinitions

SsimParameters::SsimParameters() : _WindowWidth(0), _WindowHeight(0), _WindowDepth(0) { }

int Compare::measureId;

BeginPropertyDefinitions(Compare)

  ReflectableBase(DefaultWorkflowElement<Compare>)

  WorkflowProperty(Image1, Input("I1"), NotNull<Type>())
  WorkflowProperty(Image2, Input("I2"), NotNull<Type>())
  WorkflowProperty(Measure, Enumerator<Type>(), Dummy(measureId = Id))
  WorkflowProperty(Parameters, Reflectable<Type>())
  WorkflowProperty(Value, Output(""))

EndPropertyDefinitions

Compare::Compare() {
  setLabel("MSE");

  Changed.connect(EventHandler<Compare>(this, &Compare::changedHandler));
}

void Compare::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == measureId) {
    switch (getMeasure()) {
    case SimilarityMeasure::MSE:
      if (!boost::dynamic_pointer_cast<NoMeasureParameters>(getParameters()))
        setParameters(boost::make_shared<NoMeasureParameters>());
      break;

    case SimilarityMeasure::SSIM:
      if (!boost::dynamic_pointer_cast<SsimParameters>(getParameters()))
        setParameters(boost::make_shared<SsimParameters>());
      break;
    }
  }
}

CompareChecker compareChecker;

} /* namespace imageprocessing */

} /* namespace gml */
