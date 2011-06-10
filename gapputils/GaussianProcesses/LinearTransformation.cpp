/*
 * LinearTransformation.cpp
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#include "LinearTransformation.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(LinearTransformation)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Transpose, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OriginalFeatureCount, Input("InFC"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ReducedFeatureCount, Input(), Output("OutFC"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleCount, Input("SC"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Transformation, Input("M"), Volatile(), Hide(), NotEqual<double*>(0), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Input, Input("In"), Volatile(), Hide(), NotEqual<double*>(0), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Output, Output("Out"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

LinearTransformation::LinearTransformation() : _Transformation(0), _Input(0), _Output(0), data(0) {
  WfeUpdateTimestamp
  setLabel("LinearTransformation");

  Changed.connect(capputils::EventHandler<LinearTransformation>(this, &LinearTransformation::changedHandler));
}

LinearTransformation::~LinearTransformation() {
  if (data)
    delete data;
}

void LinearTransformation::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void LinearTransformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new LinearTransformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  double* output = new double[getReducedFeatureCount() * getSampleCount()];

  lintrans(output, getReducedFeatureCount(), getSampleCount(), getInput(), getOriginalFeatureCount(), getTransformation(), getTranspose());

  if (data->getOutput())
    delete (data->getOutput());
  data->setOutput(output);
}

void LinearTransformation::writeResults() {
  if (!data)
    return;

  if (getOutput())
    delete getOutput();
  setOutput(data->getOutput());
  data->setOutput(0);
}

}
