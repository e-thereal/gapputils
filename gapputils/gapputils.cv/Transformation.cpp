/*
 * Transformation.cpp
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#include "Transformation.h"

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

#include <culib/transform.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Transformation)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(XTrans, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(YTrans, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Matrix, Output("M"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Transformation::Transformation() : _XTrans(0), _YTrans(0), data(0) {
  WfeUpdateTimestamp
  setLabel("Transformation");

  Changed.connect(capputils::EventHandler<Transformation>(this, &Transformation::changedHandler));
}

Transformation::~Transformation() {
  if (data)
    delete data;
}

void Transformation::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Transformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Transformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  data->setMatrix(boost::shared_ptr<fmatrix4>(new fmatrix4(make_fmatrix4_translation(getXTrans(), getYTrans()))));
}

void Transformation::writeResults() {
  if (!data)
    return;

  setMatrix(data->getMatrix());
}

}

}
