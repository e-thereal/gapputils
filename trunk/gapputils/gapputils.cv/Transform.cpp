/*
 * Transform.cpp
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#include "Transform.h"

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

#include "cuda_util.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Transform)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("In"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Matrix, Input("M"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Width, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Out"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Transform::Transform() : _Width(0), _Height(0), data(0) {
  WfeUpdateTimestamp
  setLabel("Transform");

  Changed.connect(capputils::EventHandler<Transform>(this, &Transform::changedHandler));
}

Transform::~Transform() {
  if (data)
    delete data;
}

void Transform::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Transform::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Transform();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage() || !getMatrix())
    return;

  boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImage());
  boost::shared_ptr<image_t> output(new image_t(getWidth(), getHeight(), 1));
  boost::shared_ptr<culib::ICudaImage> cuoutput = make_cuda_image(*output);

  culib::transform3D(cuoutput->getDevicePointer(), input->getCudaArray(), cuoutput->getSize(), *getMatrix());
  cuoutput->saveDeviceToOriginalImage();

  data->setOutputImage(output);
}

void Transform::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
