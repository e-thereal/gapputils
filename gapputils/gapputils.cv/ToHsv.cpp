#include "ToHsv.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <culib/CudaImage.h>

#include <gapputils/HideAttribute.h>

#include <QColor>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ToHsv)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Hue, Output("H"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Saturation, Output("S"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Value, Output("V"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ToHsv::ToHsv(void) : data(0)
{
  setLabel("ToHsv");
  Changed.connect(capputils::EventHandler<ToHsv>(this, &ToHsv::changedEventHandler));
}

ToHsv::~ToHsv(void)
{
  if (data)
    delete data;
}

void ToHsv::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

void ToHsv::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace culib;

  if (!data)
    data = new ToHsv();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getImagePtr())
    return;

  boost::shared_ptr<QImage> image = getImagePtr();
  const int width = image->width();
  const int height = image->height();
 
  boost::shared_ptr<CudaImage> hueImage(new CudaImage(dim3(width, height)));
  boost::shared_ptr<CudaImage> saturationImage(new CudaImage(dim3(width, height)));
  boost::shared_ptr<CudaImage> valueImage(new CudaImage(dim3(width, height)));
  float* hueBuffer = hueImage->getOriginalImage();
  float* saturationBuffer = saturationImage->getOriginalImage();
  float* valueBuffer = valueImage->getOriginalImage();

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      QColor color(image->pixel(x, y));
      qreal h, s, v;
      color.toHsv().getHsvF(&h, &s, &v);
      hueBuffer[i] = (float)h;
      saturationBuffer[i] = (float)s;
      valueBuffer[i] = (float)v;
    }
  }
  hueImage->resetWorkingCopy();
  saturationImage->resetWorkingCopy();
  valueImage->resetWorkingCopy();

  data->setHue(hueImage);
  data->setSaturation(saturationImage);
  data->setValue(valueImage);
}

void ToHsv::writeResults() {
  if (!data)
    return;

  setHue(data->getHue());
  setSaturation(data->getSaturation());
  setValue(data->getValue());
}

}

}