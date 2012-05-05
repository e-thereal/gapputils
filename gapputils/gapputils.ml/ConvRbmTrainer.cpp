/*
 * ConvRbmTrainer.cpp
 *
 *  Created on: Mar 5, 2012
 *      Author: tombr
 */

#include "ConvRbmTrainer.h"

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

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(ConvRbmTrainer)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InitialModel, Input("CRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Tensors, Input("Imgs"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(SampleVisibles, Observe(PROPERTY_ID))
  DefineProperty(EpochCount, Observe(PROPERTY_ID))
  DefineProperty(BatchSize, Observe(PROPERTY_ID))
  DefineProperty(LearningRate, Observe(PROPERTY_ID))
  DefineProperty(SparsityTarget, Observe(PROPERTY_ID))
  DefineProperty(SparsityPenalty, Observe(PROPERTY_ID))
  DefineProperty(UseRandomSamples, Observe(PROPERTY_ID))
  DefineProperty(CalculateBaseline, Observe(PROPERTY_ID))
  DefineProperty(ShowProgress, Observe(PROPERTY_ID))
  DefineProperty(Model, Output("CRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Filters, Output("F"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ConvRbmTrainer::ConvRbmTrainer()
 : _SampleVisibles(false), _EpochCount(50), _BatchSize(10), _LearningRate(0.01),
   _SparsityTarget(0.001), _SparsityPenalty(0.01), _UseRandomSamples(false),
   _CalculateBaseline(false), _ShowProgress(false), data(0)
{
  WfeUpdateTimestamp
  setLabel("ConvRbmTrainer");

  Changed.connect(capputils::EventHandler<ConvRbmTrainer>(this, &ConvRbmTrainer::changedHandler));
}

ConvRbmTrainer::~ConvRbmTrainer() {
  if (data)
    delete data;
}

void ConvRbmTrainer::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl

void ConvRbmTrainer::writeResults() {
//  std::cout << "Host:" << std::endl;
//  ConvRbmTrainer test;
//  LOCATE(test, InitialModel);
//  LOCATE(test, Tensors);
//  LOCATE(test, SampleVisibles);
//  LOCATE(test, EpochCount);
//  LOCATE(test, BatchSize);
//  LOCATE(test, LearningRate);
//  LOCATE(test, Model);

  if (!data)
    return;

  setModel(data->getModel());
  setFilters(data->getFilters());
}

}

}
