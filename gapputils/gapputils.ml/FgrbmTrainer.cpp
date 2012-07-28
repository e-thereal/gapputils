/*
 * FgrbmTrainer.cpp
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#include "FgrbmTrainer.h"

#include <capputils/EventHandler.h>
#include <capputils/FlagAttribute.h>
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

  DefineProperty(ConditionalsVector, Input("X"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisiblesVector, Input("Y"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(InitialFgrbmModel, Input("FGRBM"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(SampleVisibles, Flag(), Observe(Id), TimeStamp(Id))
  DefineProperty(EpochCount, Observe(Id), TimeStamp(Id))
  DefineProperty(BatchSize, Observe(Id), TimeStamp(Id))
  DefineProperty(LearningRate, Observe(Id), TimeStamp(Id))
  DefineProperty(FgrbmModel, Output("FGRBM"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(Wx, Output(), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(Wy, Output(), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

FgrbmTrainer::FgrbmTrainer()
 : _SampleVisibles(false), _EpochCount(30), _BatchSize(1), _LearningRate(1e-3), data(0)
{
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
