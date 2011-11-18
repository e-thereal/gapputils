/*
 * RbmWriter.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "RbmWriter.h"

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

BeginPropertyDefinitions(RbmWriter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(RbmModel, Input("RBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Filename, Output("File"), Filename("RBM Model (*.rbm)"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmWriter::RbmWriter() : data(0) {
  WfeUpdateTimestamp
  setLabel("RbmWriter");

  Changed.connect(capputils::EventHandler<RbmWriter>(this, &RbmWriter::changedHandler));
}

RbmWriter::~RbmWriter() {
  if (data)
    delete data;
}

void RbmWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmWriter::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
