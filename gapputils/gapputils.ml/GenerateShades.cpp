/*
 * GenerateShades.cpp
 *
 *  Created on: Jan 30, 2012
 *      Author: tombr
 */

#include "GenerateShades.h"

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

#include <algorithm>

#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(GenerateShades)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("Img"), Description("Must be a single slice image."), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(MaxSummand, Observe(PROPERTY_ID))
  DefineProperty(MinMultiplier, Observe(PROPERTY_ID))
  DefineProperty(MaxMultiplier, Observe(PROPERTY_ID))
  DefineProperty(Count, Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Description("Output images are stacked together forming a multi-slice image."), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

GenerateShades::GenerateShades() : data(0) {
  WfeUpdateTimestamp
  setLabel("GenerateShades");

  Changed.connect(capputils::EventHandler<GenerateShades>(this, &GenerateShades::changedHandler));
}

GenerateShades::~GenerateShades() {
  if (data)
    delete data;
}

void GenerateShades::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

template<class T>
class axpy_34 {
private:
  T a, y;

public:
  axpy_34(const T& a, const T& y) : a(a), y(y) { }

  T operator()(const T& x) const {
    return a * x + y;
  }
};

void GenerateShades::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using capputils::Severity;
  capputils::Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Warning);

  if (!data)
    data = new GenerateShades();

  if (!capputils::Verifier::Valid(*this, dlog))
    return;

  if (!getInputImage()) {
    dlog() << "No input image given. Abbording.";
    return;
  }

  image_t& input = *getInputImage();

  if (input.getSize()[2] != 1) {
    dlog() << "Input image must contain exactly one slice. Abbording.";
    return;
  }

  image_t::dim_t newSize;
  newSize[0] = input.getSize()[0];
  newSize[1] = input.getSize()[1];
  newSize[2] = getCount();

  boost::shared_ptr<image_t> output(new image_t(newSize, input.getPixelSize()));
  float* outputBuffer = output->getData();
  float* inputBuffer = input.getData();

  const int pixelCount = newSize[0] * newSize[1];

  for (unsigned i = 0; i < getCount(); ++i) {
    float a = (float)rand() / RAND_MAX * (getMaxMultiplier() - getMinMultiplier()) + getMinMultiplier();
    float y = (float)rand() / RAND_MAX * getMaxSummand();
    std::transform(inputBuffer, inputBuffer + pixelCount, outputBuffer + pixelCount * i, axpy_34<float>(a, y));
  }

  data->setOutputImage(output);
}

void GenerateShades::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
