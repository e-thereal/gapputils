/*
 * Resample.cpp
 *
 *  Created on: Aug 31, 2011
 *      Author: tombr
 */

#include "Resample.h"

#include <capputils/DescriptionAttribute.h>
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

#include "util.h"

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
  DefineProperty(Width, Description("Absolute width after resizing no matter what the initial width has been."), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Description("Absolute height after resizing no matter what the initial height has been."), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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
    boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImage());
    boost::shared_ptr<image_t> output(new image_t(getWidth(), getHeight(), input->getSize().z));
    boost::shared_ptr<culib::ICudaImage> cuoutput = make_cuda_image(*output);
    
    fmatrix4 scalingMatrix = make_fmatrix4_scaling(
        (float)input->getSize().x / (float)cuoutput->getSize().x,
        (float)input->getSize().y / (float)cuoutput->getSize().y,
        1.f);
    culib::transform3D(cuoutput->getDevicePointer(), input->getCudaArray(), cuoutput->getSize(), scalingMatrix);
    cuoutput->saveDeviceToOriginalImage();

    data->setOutputImage(output);
  }

  if (getInputImages()) {
    boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > outputs(new std::vector<boost::shared_ptr<image_t> >());
    for (unsigned i = 0; i < getInputImages()->size(); ++i) {
      boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImages()->at(i));
      boost::shared_ptr<image_t> output(new image_t(getWidth(), getHeight(), input->getSize().z));
      boost::shared_ptr<culib::ICudaImage> cuoutput = make_cuda_image(*output);

      fmatrix4 scalingMatrix = make_fmatrix4_scaling(
          (float)input->getSize().x / (float)cuoutput->getSize().x,
          (float)input->getSize().y / (float)cuoutput->getSize().y,
          1.f);
      culib::transform3D(cuoutput->getDevicePointer(), input->getCudaArray(), cuoutput->getSize(), scalingMatrix);
      cuoutput->saveDeviceToOriginalImage();

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
