/*
 * RbmReader.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "RbmReader.h"

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
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Input("File"), Filename("RBM Model (*.rbm)"), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RbmModel, Output("RBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmReader::RbmReader() : _VisibleCount(0), _HiddenCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmReader");

  Changed.connect(capputils::EventHandler<RbmReader>(this, &RbmReader::changedHandler));
}

RbmReader::~RbmReader() {
  if (data)
    delete data;
}

void RbmReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmReader::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
  setVisibleCount(data->getVisibleCount());
  setHiddenCount(data->getHiddenCount());
}

}

}
