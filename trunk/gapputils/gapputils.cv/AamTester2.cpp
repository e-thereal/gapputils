/*
 * AamTester2.cpp
 *
 *  Created on: Jul 26, 2011
 *      Author: tombr
 */

#include "AamTester2.h"

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

#include "AamUtils.h"
#include "AamMatchFunction.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamTester2)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Grid, Input(), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Output("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Similarity, Output("Sim"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamTester2::AamTester2() : _Similarity(0), data(0) {
  WfeUpdateTimestamp
  setLabel("AamTester2");

  Changed.connect(capputils::EventHandler<AamTester2>(this, &AamTester2::changedHandler));
}

AamTester2::~AamTester2() {
  if (data)
    delete data;
}

void AamTester2::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamTester2::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamTester2();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getImage() || !getGrid())
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();

  boost::shared_ptr<std::vector<float> > parameters(new std::vector<float>(model->getAppearanceParameterCount()));
  std::vector<double> dshape(model->getShapeParameterCount());
  std::vector<float> fshape(model->getShapeParameterCount());

  AamUtils::getAppearanceParameters(parameters.get(), model.get(), getGrid().get(), getImage().get());
  data->setParameterVector(parameters);

  AamMatchFunction objective(getImage(), model, true, AamMatchFunction::SSD);
  AamUtils::getShapeParameters(&fshape, model.get(), parameters.get());
  copy(fshape.begin(), fshape.end(), dshape.begin());
  data->setSimilarity(objective.eval(dshape));
}

void AamTester2::writeResults() {
  if (!data)
    return;

  setParameterVector(data->getParameterVector());
  setSimilarity(data->getSimilarity());
}

}

}
