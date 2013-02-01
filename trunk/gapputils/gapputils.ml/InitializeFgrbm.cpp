/*
 * InitializeFgrbm.cpp
 *
 *  Created on: Jan 25, 2012
 *      Author: tombr
 */

#include "InitializeFgrbm.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/FlagAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(InitializeFgrbm)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(ConditionalsVector, Input("X"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisiblesVector, Input("Y"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisibleCount, Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenCount, Observe(Id), TimeStamp(Id))
  DefineProperty(FactorCount, Observe(Id), TimeStamp(Id))
  DefineProperty(WeightStddevs, Observe(Id))
  DefineProperty(DiagonalWeightMeans, Observe(Id))
  DefineProperty(InitialHidden, Observe(Id), TimeStamp(Id))
  DefineProperty(IsGaussian, Flag(), Observe(Id), TimeStamp(Id))

  DefineProperty(FgrbmModel, Output("FGRBM"), Volatile(), ReadOnly(), Observe(Id))

EndPropertyDefinitions

#define LOCATE(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

InitializeFgrbm::InitializeFgrbm()
 : _VisibleCount(1), _HiddenCount(100), _FactorCount(200), _WeightStddevs(0.01),
   _DiagonalWeightMeans(0), _InitialHidden(0), _IsGaussian(true), data(0)
{
  setLabel("InitializeFgrbm");

  Changed.connect(capputils::EventHandler<InitializeFgrbm>(this, &InitializeFgrbm::changedHandler));
}

InitializeFgrbm::~InitializeFgrbm() {
  if (data)
    delete data;
}

void InitializeFgrbm::changedHandler(capputils::ObservableClass* sender, int eventId) {
}

void InitializeFgrbm::writeResults() {
  if (!data)
    return;

  setFgrbmModel(data->getFgrbmModel());
}

}

}
