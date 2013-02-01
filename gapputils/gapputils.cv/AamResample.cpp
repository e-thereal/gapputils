/*
 * AamResample.cpp
 *
 *  Created on: Sep 2, 2011
 *      Author: tombr
 */

#include "AamResample.h"

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

#include <capputils/HideAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamResample)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputModel, Input("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(OutputModel, Output("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(Width, Observe(Id), TimeStamp(Id))
  DefineProperty(Height, Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

AamResample::AamResample() : _ColumnCount(4), _RowCount(4), _Width(128), _Height(128), data(0) {
  WfeUpdateTimestamp
  setLabel("AamResample");

  Changed.connect(capputils::EventHandler<AamResample>(this, &AamResample::changedHandler));
}

AamResample::~AamResample() {
  if (data)
    delete data;
}

void AamResample::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {

}

void AamResample::execute(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  if (!data)
    data = new AamResample();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputModel())
    return;

  boost::shared_ptr<ActiveAppearanceModel> input = getInputModel();
  boost::shared_ptr<ActiveAppearanceModel> output(new ActiveAppearanceModel());



  data->setOutputModel(output);
}

void AamResample::writeResults() {
  if (!data)
    return;

  setOutputModel(data->getOutputModel());
}

}

}
