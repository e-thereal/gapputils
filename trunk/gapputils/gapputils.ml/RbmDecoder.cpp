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

  if (!getRbmModel() || !getHiddenVector() || (getHiddenVector()->size() % (getRbmModel()->getHiddenCount() + 1)))
      return;

  const unsigned visibleCount = getRbmModel()->getVisibleCount();
  const unsigned hiddenCount = getRbmModel()->getHiddenCount();

  const unsigned sampleCount = getHiddenVector()->size() / (hiddenCount + 1);
  boost::shared_ptr<std::vector<float> > visibleVector(new std::vector<float>(sampleCount * (visibleCount + 1)));

  // visibleVector = sigm(WH')', since lintrans is column major, it automatically computes transposes
  culib::lintrans(&visibleVector->at(0), &getRbmModel()->getWeightMatrix()->at(0), &getHiddenVector()->at(0),
      hiddenCount + 1, sampleCount, visibleCount + 1, true);

  if (!getGaussianModel())
    std::transform(visibleVector->begin(), visibleVector->end(), visibleVector->begin(), sigmoid);

  // Decode approximation (reconstruction)
  data->setVisibleVector(getRbmModel()->decodeApproximation(visibleVector.get()));
}

void RbmDecoder::writeResults() {
  if (!data)
    return;

  setVisibleVector(data->getVisibleVector());
}

}

}
