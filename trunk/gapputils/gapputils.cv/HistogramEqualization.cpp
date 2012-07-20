#include "HistogramEqualization.h"

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

#include <culib/CudaImage.h>
#include <culib/filter.h>

#include <gapputils/HideAttribute.h>

#include "cuda_util.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace culib;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(HistogramEqualization)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BinCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

HistogramEqualization::HistogramEqualization() : _BinCount(256), data(0) {
  WfeUpdateTimestamp
  setLabel("HistogramEqualization");

  Changed.connect(capputils::EventHandler<HistogramEqualization>(this, &HistogramEqualization::changedHandler));
}

HistogramEqualization::~HistogramEqualization() {
  if (data)
    delete data;
}

void HistogramEqualization::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void HistogramEqualization::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new HistogramEqualization();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage())
    return;

  boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImage());
  boost::shared_ptr<image_t> output(new image_t(getInputImage()->getSize(), getInputImage()->getPixelSize()));

  culib::CudaImage cuoutput(input->getSize(), input->getVoxelSize());
  culib::equalizeHistogram(cuoutput.getDevicePointer(), input->getDevicePointer(), input->getSize(), getBinCount(), 1.f/(float)getBinCount(), false);
  cuoutput.saveDeviceToOriginalImage();
  data->setOutputImage(output);
}

void HistogramEqualization::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
