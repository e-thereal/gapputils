/*
 * FeatureCollages.cpp
 *
 *  Created on: Apr 3, 2012
 *      Author: tombr
 */

#include "FeatureCollages.h"

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

#include <algorithm>
#include <cstdlib>

#include <boost/lambda/lambda.hpp>

#include <culib/CudaImage.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

DefineEnum(ImageFusion);

BeginPropertyDefinitions(FeatureCollages)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImages, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(FeatureCount, Observe(PROPERTY_ID))
  DefineProperty(ImageCount, Observe(PROPERTY_ID))
  ReflectableProperty(Fusion, Observe(PROPERTY_ID))
  DefineProperty(OutputImages, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Debug"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

FeatureCollages::FeatureCollages() : data(0) {
  WfeUpdateTimestamp
  setLabel("FeatureCollages");

  Changed.connect(capputils::EventHandler<FeatureCollages>(this, &FeatureCollages::changedHandler));
}

FeatureCollages::~FeatureCollages() {
  if (data)
    delete data;
}

void FeatureCollages::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FeatureCollages::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace boost::lambda;

  if (!data)
    data = new FeatureCollages();

  if (!capputils::Verifier::Valid(*this))
    return;

  std::vector<boost::shared_ptr<culib::ICudaImage> >& primitives = *getInputImages();

  const int width = primitives[0]->getSize().x;
  const int height = primitives[0]->getSize().y;
  const int count = width * height;

  boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > > outputs(
      new std::vector<boost::shared_ptr<culib::ICudaImage> >());

  for (int iImage = 0; iImage < getImageCount(); ++iImage) {
    boost::shared_ptr<culib::CudaImage> output(new culib::CudaImage(dim3(width, height)));
    float* buffer = output->getWorkingCopy();

    switch (getFusion()) {
    case ImageFusion::Addition:
      std::fill(buffer, buffer + count, 0.f);
      break;

    case ImageFusion::Multiplication:
      std::fill(buffer, buffer + count, 1.f);
      break;
    }

    for (int iPrimitive = 0; iPrimitive < getFeatureCount(); ++iPrimitive) {
      culib::ICudaImage& primitive = *primitives[rand() % primitives.size()];
      switch(getFusion()) {
      case ImageFusion::Addition:
        std::transform(buffer, buffer + count, primitive.getWorkingCopy(), buffer, _1 + _2);
        break;

      case ImageFusion::Multiplication:
        std::transform(buffer, buffer + count, primitive.getWorkingCopy(), buffer, _1 * _2);
        break;
      }

    }
    outputs->push_back(output);
    data->setOutputImage(output);
  }
  data->setOutputImages(outputs);
}

void FeatureCollages::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
  setOutputImages(data->getOutputImages());
}

}

}
