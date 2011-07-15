/*
 * AamFitter.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamFitter.h"

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

#include <optlib/DownhillSimplexOptimizer.h>

#include "AamMatchFunction.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamFitter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Output("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamFitter::AamFitter() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamFitter");

  Changed.connect(capputils::EventHandler<AamFitter>(this, &AamFitter::changedHandler));
}

AamFitter::~AamFitter() {
  if (data)
    delete data;
}

void AamFitter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamFitter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamFitter();

  if (!capputils::Verifier::Valid(*this))
    return;

  // Test something in here
  // Use generator to generate images from the model
  // evaluate model fit using altered parameter vectors
  // Try to find optimal parameters using different optimization algorithms
  // compare with known solution

  AamMatchFunction objective(getInputImage(), getActiveAppearanceModel());
  optlib::DownhillSimplexOptimizer optimizer;
  std::vector<double> parameters(getActiveAppearanceModel()->getModelParameterCount());
  optimizer.minimize(parameters, objective);
}

void AamFitter::writeResults() {
  if (!data)
    return;

}

}

}
