/*
 * ResampleFgrbmModel.cpp
 *
 *  Created on: Feb 1, 2012
 *      Author: tombr
 */

#include "ResampleFgrbmModel.h"

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

BeginPropertyDefinitions(ResampleFgrbmModel)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputModel, Input("FGRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(InputWidth, Observe(PROPERTY_ID))
  DefineProperty(InputHeight, Observe(PROPERTY_ID))
  DefineProperty(OutputWidth, Observe(PROPERTY_ID))
  DefineProperty(OutputHeight, Observe(PROPERTY_ID))
  DefineProperty(OutputModel, Output("FGRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ResampleFgrbmModel::ResampleFgrbmModel() : data(0) {
  WfeUpdateTimestamp
  setLabel("ResampleFgrbmModel");

  Changed.connect(capputils::EventHandler<ResampleFgrbmModel>(this, &ResampleFgrbmModel::changedHandler));
}

ResampleFgrbmModel::~ResampleFgrbmModel() {
  if (data)
    delete data;
}

void ResampleFgrbmModel::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ResampleFgrbmModel::writeResults() {
  if (!data)
    return;

  setOutputModel(data->getOutputModel());
}

}

}
