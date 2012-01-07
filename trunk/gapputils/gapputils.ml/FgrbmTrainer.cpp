/*
 * FgrbmTrainer.cpp
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#include "FgrbmTrainer.h"

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

BeginPropertyDefinitions(FgrbmTrainer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(ConditionalsVector, Input("X"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisiblesVector, Input("Y"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FgrbmModel, Output("FGRBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FactorCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleVisibles, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(EpochCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BatchSize, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LearningRate, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InitialHidden, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Wx, Output(), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Wy, Output(), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FgrbmTrainer::FgrbmTrainer() : data(0) {
  WfeUpdateTimestamp
  setLabel("FgrbmTrainer");

  Changed.connect(capputils::EventHandler<FgrbmTrainer>(this, &FgrbmTrainer::changedHandler));
}

FgrbmTrainer::~FgrbmTrainer() {
  if (data)
    delete data;
}

void FgrbmTrainer::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FgrbmTrainer::writeResults() {
  if (!data)
    return;

  setFgrbmModel(data->getFgrbmModel());
  setWx(data->getWx());
  setWy(data->getWy());
}

}

}
