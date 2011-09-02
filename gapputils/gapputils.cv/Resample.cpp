/*
 * Resample.cpp
 *
 *  Created on: Aug 31, 2011
 *      Author: tombr
 */

#include "Resample.h"

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
#include <culib/transform.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Resample)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(InputImage, Input("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImages, Input("Imgs"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImages, Output("Imgs"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Width, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Resample::Resample() : _Width(100), _Height(100), data(0) {
  WfeUpdateTimestamp
  setLabel("Resample");

  Changed.connect(capputils::EventHandler<Resample>(this, &Resample::changedHandler));
}

Resample::~Resample() {
  if (data)
    delete data;
}

void Resample::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Resample::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Resample();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (getInputImage()) {
    culib::ICudaImage* input = getInputImage().get();
    boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(getWidth(), getHeight())));
    fmatrix4 scalingMatrix = make_fmatrix4_scaling(
        (float)input->getSize().x / (float)output->getSize().x,
        (float)input->getSize().y / (float)output->getSize().y,
        1.f);
    culib::transform3D(output->getDevicePointer(), input->getCudaArray(), output->getSize(), scalingMatrix);
    output->saveDeviceToWorkingCopy();
    input->freeCaches();
    output->freeCaches();

    data->setOutputImage(output);
  }

  if (getInputImages()) {
    boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > > outputs(new std::vector<boost::shared_ptr<culib::ICudaImage> >());
    for (unsigned i = 0; i < getInputImages()->size(); ++i) {
      culib::ICudaImage* input = getInputImages()->at(i).get();
      boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(getWidth(), getHeight())));
      fmatrix4 scalingMatrix = make_fmatrix4_scaling(
          (float)input->getSize().x / (float)output->getSize().x,
          (float)input->getSize().y / (float)output->getSize().y,
          1.f);
      culib::transform3D(output->getDevicePointer(), input->getCudaArray(), output->getSize(), scalingMatrix);
      output->saveDeviceToWorkingCopy();
      input->freeCaches();
      output->freeCaches();

      outputs->push_back(output);
    }
    data->setOutputImages(outputs);
  }
}

void Resample::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
  setOutputImages(data->getOutputImages());
}

}

}
