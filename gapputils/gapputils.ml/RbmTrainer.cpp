/*
 * RbmTrainer.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmTrainer.h"

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

  DefineProperty(TrainingSet, Input("Data"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RbmModel, Output("RBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(EpochCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BatchSize, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LearningRate, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InitialHidden, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SparsityTarget, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SparsityWeight, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  //DefineProperty(PosData, Output("PD"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  //DefineProperty(NegData, Output("ND"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmTrainer::RbmTrainer()
 : _VisibleCount(1), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01f), _InitialHidden(0.f),
   _SparsityTarget(0.1f), _SparsityWeight(0.1f), _IsGaussian(false), data(0)
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
  //setNegData(data->getNegData());
  //setPosData(data->getPosData());
}

}

}
