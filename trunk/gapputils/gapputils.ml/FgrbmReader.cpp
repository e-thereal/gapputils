/*
 * FgrbmReader.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "FgrbmReader.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Serializer.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FgrbmReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Input("File"), Filename("FGRBM Model (*.fgrbm)"), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FgrbmModel, Output("FGRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FgrbmReader::FgrbmReader() : _VisibleCount(0), _HiddenCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("FgrbmReader");

  Changed.connect(capputils::EventHandler<FgrbmReader>(this, &FgrbmReader::changedHandler));
}

FgrbmReader::~FgrbmReader() {
  if (data)
    delete data;
}

void FgrbmReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FgrbmReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FgrbmReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<FgrbmModel> fgrbm(new FgrbmModel());
  capputils::Serializer::readFromFile(*fgrbm, getFilename());

  data->setFgrbmModel(fgrbm);
  data->setVisibleCount(fgrbm->getVisibleBiases()->size());
  data->setHiddenCount(fgrbm->getHiddenBiases()->size());
}

void FgrbmReader::writeResults() {
  if (!data)
    return;

  setFgrbmModel(data->getFgrbmModel());
  setVisibleCount(data->getVisibleCount());
  setHiddenCount(data->getHiddenCount());
}

}

}
