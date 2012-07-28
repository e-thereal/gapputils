/*
 * RbmTrainer.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmTrainer.h"

#include <capputils/DescriptionAttribute.h>
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

#include <culib/lintrans.h>

#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

#include <boost/progress.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include "RbmEncoder.h"
#include "RbmDecoder.h"
#include "ublas_io.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmTrainer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(TrainingSet, Input("Data"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(RbmModel, Output("RBM"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisibleCount, Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenCount, Observe(Id), TimeStamp(Id))
  DefineProperty(SampleHiddens, Observe(Id), TimeStamp(Id))
  DefineProperty(EpochCount, Observe(Id), TimeStamp(Id))
  DefineProperty(BatchSize, Observe(Id), TimeStamp(Id))
  DefineProperty(LearningRate, Observe(Id), TimeStamp(Id))
  DefineProperty(InitialHidden, Observe(Id), TimeStamp(Id))
  DefineProperty(SparsityTarget, Observe(Id), TimeStamp(Id))
  DefineProperty(SparsityWeight, Observe(Id), TimeStamp(Id))
  DefineProperty(IsGaussian, Observe(Id), TimeStamp(Id))
  DefineProperty(Weights, Output("W"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(ShowWeights, Description("Only the first ShowWeights features are shown."), Observe(Id))
  DefineProperty(ShowEvery, Description("Debug output is shown only every ShowEvery epochs."), Observe(Id))
  //DefineProperty(PosData, Output("PD"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  //DefineProperty(NegData, Output("ND"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

RbmTrainer::RbmTrainer()
 : _VisibleCount(1), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01f), _InitialHidden(0.f),
   _SparsityTarget(0.1f), _SparsityWeight(0.1f), _IsGaussian(false), _ShowWeights(0), _ShowEvery(1), data(0)
{
  WfeUpdateTimestamp
  setLabel("RbmTrainer");

  Changed.connect(capputils::EventHandler<RbmTrainer>(this, &RbmTrainer::changedHandler));
}

RbmTrainer::~RbmTrainer() {
  if (data)
    delete data;
}

void RbmTrainer::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

template <class T>
T square(const T& a) { return a * a; }

void RbmTrainer::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
  setWeights(data->getWeights());
  //setNegData(data->getNegData());
  //setPosData(data->getPosData());
}

}

}
