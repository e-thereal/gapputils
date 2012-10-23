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

  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<RbmTrainer>)

  WorkflowProperty(TrainingSet, Input("Data"), NotNull<Type>())
  WorkflowProperty(RbmModel, Output("RBM"))
  WorkflowProperty(VisibleCount)
  WorkflowProperty(HiddenCount)
  WorkflowProperty(SampleHiddens)
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(InitialHidden)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight)
  WorkflowProperty(IsGaussian)
  WorkflowProperty(MakeBernoulli)
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(BernoulliData, Output("B"))
  WorkflowProperty(ShowWeights, Description("Only the first ShowWeights features are shown."))
  WorkflowProperty(ShowEvery, Description("Debug output is shown only every ShowEvery epochs."))
  //DefineProperty(PosData, Output("PD"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  //DefineProperty(NegData, Output("ND"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

RbmTrainer::RbmTrainer()
 : _VisibleCount(1), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01f), _InitialHidden(0.f),
   _SparsityTarget(0.1f), _SparsityWeight(0.1f), _IsGaussian(false),
   _MakeBernoulli(false), _ShowWeights(0), _ShowEvery(1)
{
  setLabel("RbmTrainer");
}

RbmTrainer::~RbmTrainer() {
}

template <class T>
T square(const T& a) { return a * a; }

}

}
