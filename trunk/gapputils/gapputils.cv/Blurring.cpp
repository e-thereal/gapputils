/*
 * Blurring.cpp
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#include "Blurring.h"

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

#include <culib/CudaImage.h>
#include <regutil/CudaImage.h>

#include <algorithm>

#include "util.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Blurring)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Sigma, Observe(PROPERTY_ID))
  DefineProperty(InPlane, Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Blurring::Blurring() : _Sigma(5.f), _InPlane(true), data(0) {
  WfeUpdateTimestamp
  setLabel("Blurring");

  Changed.connect(capputils::EventHandler<Blurring>(this, &Blurring::changedHandler));
}

Blurring::~Blurring() {
  if (data)
    delete data;
}

void Blurring::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Blurring::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Blurring();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage())
    return;

  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));
  
  if (getInPlane()) {
    const int count = input.getSize()[0] * input.getSize()[1];
    for (int i = 0; i < input.getSize()[2]; ++i) {
      regutil::CudaImage inputImage(dim3(input.getSize()[0], input.getSize()[1]),
          dim3(input.getPixelSize()[0], input.getPixelSize()[1], input.getPixelSize()[2]), input.getData() + i * count);
      inputImage.blurImage(getSigma());
      inputImage.saveDeviceToWorkingCopy();
      std::copy(inputImage.getWorkingCopy(), inputImage.getWorkingCopy() + count, output->getData() + i * count);
    }
  } else {
    const int count = input.getSize()[0] * input.getSize()[1] * input.getSize()[2];
    regutil::CudaImage inputImage(dim3(input.getSize()[0], input.getSize()[1], input.getSize()[2]),
        dim3(input.getPixelSize()[0], input.getPixelSize()[1], input.getPixelSize()[2]), input.getData());
    inputImage.blurImage(getSigma());
    inputImage.saveDeviceToWorkingCopy();
    std::copy(inputImage.getWorkingCopy(), inputImage.getWorkingCopy() + count, output->getData());
  }
  
  data->setOutputImage(output);
}

void Blurring::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
