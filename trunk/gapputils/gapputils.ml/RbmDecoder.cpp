/*
 * RbmDecoder.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmDecoder.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmDecoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(RbmModel, Input("RBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenVector, Input("In"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleVector, Output("Out"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(GaussianModel, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmDecoder::RbmDecoder() : _GaussianModel(0), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmDecoder");

  Changed.connect(capputils::EventHandler<RbmDecoder>(this, &RbmDecoder::changedHandler));
}

RbmDecoder::~RbmDecoder() {
  if (data)
    delete data;
}

void RbmDecoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmDecoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmDecoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getHiddenVector())
      return;

  //boost::shared_ptr<std::vector<float> > visibleVector = getExpectations(getHiddenVector().get(), getRbmModel().get(), getGaussianModel());

  // Decode approximation (reconstruction)
  //data->setVisibleVector(getRbmModel()->decodeApproximation(visibleVector.get()));
}

void RbmDecoder::writeResults() {
  if (!data)
    return;

  setVisibleVector(data->getVisibleVector());
}

//boost::shared_ptr<std::vector<float> > RbmDecoder::getExpectations(std::vector<float>* hiddenVector, RbmModel* rbm, bool gaussianModel) {
//  const unsigned visibleCount = rbm->getVisibleCount();
//  const unsigned hiddenCount = rbm->getHiddenCount();
//
//  const unsigned sampleCount = hiddenVector->size() / (hiddenCount + 1);
//  boost::shared_ptr<std::vector<float> > visibleVector(new std::vector<float>(sampleCount * (visibleCount + 1)));
//
//  // visibleVector = sigm(WH')', since lintrans is column major, it automatically computes transposes
//  culib::lintrans(&visibleVector->at(0), &rbm->getWeightMatrix()->at(0), &hiddenVector->at(0),
//      hiddenCount + 1, sampleCount, visibleCount + 1, true);
//
//  if (!gaussianModel)
//    std::transform(visibleVector->begin(), visibleVector->end(), visibleVector->begin(), sigmoid);
//
//  return visibleVector;
//}

//boost::shared_ptr<std::vector<float> > RbmDecoder::sampleVisibles(std::vector<float>* means, bool gaussianModel) {
//  boost::shared_ptr<std::vector<float> > visibleVector(new std::vector<float> (means->size()));
//
//  if (gaussianModel)
//    std::transform(means->begin(), means->end(), visibleVector->begin(), createNormalSample());
//  else
//    std::transform(means->begin(), means->end(), visibleVector->begin(), createBernoulliSample());
//
//  return visibleVector;
//}

//boost::shared_ptr<std::vector<float> > RbmDecoder::sampleVisibles(std::vector<float>* hiddenVector, RbmModel* rbm, bool gaussianModel) {
//  boost::shared_ptr<std::vector<float> > means = getExpectations(hiddenVector, rbm, gaussianModel);
//  return sampleVisibles(means.get(), gaussianModel);
//}

}

}
