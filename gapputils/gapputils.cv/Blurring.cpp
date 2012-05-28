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

  culib::ICudaImage& input = *getInputImage();
  boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(input.getSize(), input.getVoxelSize()));
  
  if (getInPlane()) {
    const int count = input.getSize().x * input.getSize().y;
    for (int i = 0; i < input.getSize().z; ++i) {
      regutil::CudaImage inputImage(dim3(input.getSize().x, input.getSize().y), input.getVoxelSize(), input.getWorkingCopy() + i * count);
      inputImage.blurImage(getSigma());
      inputImage.saveDeviceToWorkingCopy();
      std::copy(inputImage.getWorkingCopy(), inputImage.getWorkingCopy() + count, output->getWorkingCopy() + i * count);
    }
  } else {
    const int count = input.getSize().x * input.getSize().y * input.getSize().z;
    regutil::CudaImage inputImage(input.getSize(), input.getVoxelSize(), input.getWorkingCopy());
    inputImage.blurImage(getSigma());
    inputImage.saveDeviceToWorkingCopy();
    std::copy(inputImage.getWorkingCopy(), inputImage.getWorkingCopy() + count, output->getWorkingCopy());
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
