/*
 * SlidingWindowFilter.cpp
 *
 *  Created on: May 30, 2012
 *      Author: tombr
 */

#include "SlidingWindowFilter.h"

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

namespace cv {

BeginPropertyDefinitions(SlidingWindowFilter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  ReflectableProperty(Filter, Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

SlidingWindowFilter::SlidingWindowFilter() : data(0) {
  WfeUpdateTimestamp
  setLabel("SWF");

  Changed.connect(capputils::EventHandler<SlidingWindowFilter>(this, &SlidingWindowFilter::changedHandler));
}

SlidingWindowFilter::~SlidingWindowFilter() {
  if (data)
    delete data;
}

void SlidingWindowFilter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void SlidingWindowFilter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new SlidingWindowFilter();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void SlidingWindowFilter::writeResults() {
  if (!data)
    return;

}

}

}
