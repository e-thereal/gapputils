/*
 * RbmEncoder.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmEncoder.h"

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

#include <algorithm>
#include <random>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmEncoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(RbmModel, Input("RBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleVector, Input("In"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenVector, Output("Out"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmEncoder::RbmEncoder() : _SampleHiddens(true), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmEncoder");

  Changed.connect(capputils::EventHandler<RbmEncoder>(this, &RbmEncoder::changedHandler));
}

RbmEncoder::~RbmEncoder() {
  if (data)
    delete data;
}

void RbmEncoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmEncoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmEncoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getVisibleVector())
    return;

  // normalize visible variables -> X (design matrix with one sample per row)
//  boost::shared_ptr<std::vector<float> > visibleVector = getRbmModel()->encodeDesignMatrix(getVisibleVector().get());
//  boost::shared_ptr<std::vector<float> > hiddenVector = getExpectations(visibleVector.get(), getRbmModel().get());
//
//  // TODO: implement the sample hiddens feature
//  if (getSampleHiddens())
//    hiddenVector = sampleHiddens(hiddenVector.get());
//  data->setHiddenVector(hiddenVector);
}

void RbmEncoder::writeResults() {
  if (!data)
    return;

  setHiddenVector(data->getHiddenVector());
}

//boost::shared_ptr<std::vector<float> > RbmEncoder::getExpectations(std::vector<float>* visibleVector, RbmModel* rbm) {
//  const unsigned visibleCount = rbm->getVisibleCount();
//  const unsigned hiddenCount = rbm->getHiddenCount();
//
//  const unsigned sampleCount = visibleVector->size() / visibleCount;
//  boost::shared_ptr<std::vector<float> > hiddenVector(new std::vector<float>(sampleCount * (hiddenCount + 1)));
//
//  // hiddenVector = sigm(W'X')', since lintrans is column major, it automatically computes transposes
//  culib::lintrans(&hiddenVector->at(0), &rbm->getWeightMatrix()->at(0), &visibleVector->at(0),
//      visibleCount + 1, sampleCount, hiddenCount + 1, false);
//
//  std::transform(hiddenVector->begin(), hiddenVector->end(), hiddenVector->begin(), sigmoid);
//
//  return hiddenVector;
//}
//
//boost::shared_ptr<std::vector<float> > RbmEncoder::sampleHiddens(std::vector<float>* means) {
//  boost::shared_ptr<std::vector<float> > hiddenVector(new std::vector<float> (means->size()));
//
//  std::transform(hiddenVector->begin(), hiddenVector->end(), means->begin(), createBernoulliSample());
//
//  return hiddenVector;
//}

}

}
