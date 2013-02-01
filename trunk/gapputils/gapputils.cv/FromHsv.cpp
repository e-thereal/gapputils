#include "FromHsv.h"

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

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <QColor>
#include <cmath>

using namespace gapputils::attributes;
using namespace capputils::attributes;

namespace gapputils {

namespace cv {

int FromHsv::outputId;

BeginPropertyDefinitions(FromHsv)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Output("Img"), Volatile(), ReadOnly(), Observe(outputId = Id), TimeStamp(Id))
  DefineProperty(Hue, Input("H"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Saturation, Input("S"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Value, Input("V"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

FromHsv::FromHsv(void) : data(0)
{
  setLabel("FromHsv");
  Changed.connect(capputils::EventHandler<FromHsv>(this, &FromHsv::changedEventHandler));
}

FromHsv::~FromHsv(void)
{
  if (data)
    delete data;
}

void FromHsv::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId != outputId) {
    execute(0);
    writeResults();
  }
}

int toIntC(float value);

void FromHsv::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FromHsv();

  if (!capputils::Verifier::Valid(*this))
    return;

  image_t* hue = getHue().get();
  image_t* saturation = getSaturation().get();
  image_t* value = getValue().get();

  int width = 0;
  int height = 0;
  int pixelWidth = 1000, pixelHeight = 1000;

  if (hue) {
    width = hue->getSize()[0];
    height = hue->getSize()[1];
    pixelWidth = hue->getPixelSize()[0];
    pixelHeight = hue->getPixelSize()[1];
  } else if (saturation) {
    width = saturation->getSize()[0];
    height = saturation->getSize()[1];
    pixelWidth = saturation->getPixelSize()[0];
    pixelHeight = saturation->getPixelSize()[1];
  } else if (value) {
    width = value->getSize()[0];
    height = value->getSize()[1];
    pixelWidth = value->getPixelSize()[0];
    pixelHeight = value->getPixelSize()[1];
  }

  if (width <= 0 || height <= 0)
    return;

  if (saturation) {
    if (saturation->getSize()[0] != width || saturation->getSize()[1] != height ||
        saturation->getPixelSize()[0] != pixelWidth || saturation->getPixelSize()[1] != pixelHeight)
      return;
  }
  if (value) {
    if (value->getSize()[0] != width || value->getSize()[1] != height ||
        value->getPixelSize()[0] != pixelWidth || value->getPixelSize()[1] != pixelHeight)
      return;
  }

  boost::shared_ptr<QImage> image(new QImage(width, height, QImage::Format_ARGB32));
  image->setDotsPerMeterX(1000000/pixelWidth);
  image->setDotsPerMeterY(1000000/pixelHeight);
  float* hueBuffer = (hue ? hue->getData() : 0);
  float* saturationBuffer = (saturation ? saturation->getData() : 0);
  float* valueBuffer = (value ? value->getData() : 0);

  QColor color;
  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      float h = (hueBuffer ? hueBuffer[i] : 0);
      float s = (saturationBuffer ? saturationBuffer[i] : 0);
      float v = (valueBuffer ? valueBuffer[i] : 0);
      color.setHsvF(std::min(1.f, std::max(0.f, h)),
          std::min(1.f, std::max(0.f, s)),
          std::min(1.f, std::max(0.f, v)));
      image->setPixel(x, y, color.rgb());
    }
  }

  data->setImagePtr(image);
}

void FromHsv::writeResults() {
  if (!data)
    return;

  setImagePtr(data->getImagePtr());
}

}

}
