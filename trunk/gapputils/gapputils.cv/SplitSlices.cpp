/*
 * SplitSlices.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#include "SplitSlices.h"

#include <capputils/NotNullAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/Logbook.h>

#include <gapputils/ReadOnlyAttribute.h>

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(SplitSlices)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;

  ReflectableBase(workflow::DefaultWorkflowElement<SplitSlices>)

  DefineProperty(Volume, Input(""), Volatile(), ReadOnly(), NotNull<PROPERTY_TYPE>(), Observe(PROPERTY_ID))
  DefineProperty(Slices, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

SplitSlices::SplitSlices() {
  setLabel("Splitter");
}

SplitSlices::~SplitSlices() {
}

void SplitSlices::update(workflow::IProgressMonitor* /*monitor*/) const {
  using namespace capputils;
  Logbook& dlog = getLogbook();

  if (!getVolume()) {
    dlog(Severity::Warning) << "No input volume given. Aborting!";
    return;
  }

  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > slices(new std::vector<boost::shared_ptr<image_t> >());
  newState->setSlices(slices);
}

} /* namespace cv */

} /* namespace gapputils */
