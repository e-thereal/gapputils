/*
 * CudaImage.cpp
 *
 *  Created on: Jun 8, 2012
 *      Author: tombr
 */

#include "CudaImageInterface.h"

#include <capputils/DeprecatedAttribute.h>
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
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace host {

namespace inputs {

BeginPropertyDefinitions(CudaImage, Interface(), Deprecated("Use 'interfaces::inputs::Image' instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Value, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

CudaImage::CudaImage() : data(0) {
  WfeUpdateTimestamp
  setLabel("CudaImage");

  Changed.connect(capputils::EventHandler<CudaImage>(this, &CudaImage::changedHandler));
}

CudaImage::~CudaImage() {
  if (data)
    delete data;
}

void CudaImage::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void CudaImage::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new CudaImage();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void CudaImage::writeResults() {
  if (!data)
    return;

}

}

namespace outputs {

BeginPropertyDefinitions(CudaImage, Interface(), Deprecated("Use 'interfaces::outputs::Image' instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Value, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

CudaImage::CudaImage() : data(0) {
  WfeUpdateTimestamp
  setLabel("CudaImage");

  Changed.connect(capputils::EventHandler<CudaImage>(this, &CudaImage::changedHandler));
}

CudaImage::~CudaImage() {
  if (data)
    delete data;
}

void CudaImage::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void CudaImage::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new CudaImage();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void CudaImage::writeResults() {
  if (!data)
    return;

}

}

}

}
