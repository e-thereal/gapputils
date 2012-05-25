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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <QColor>
#include <cmath>

using namespace gapputils::attributes;

using namespace capputils::attributes;
using namespace culib;

namespace gapputils {

namespace cv {

int FromHsv::outputId;

BeginPropertyDefinitions(FromHsv)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Output("Img"), Volatile(), ReadOnly(), Observe(outputId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Hue, Input("H"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Saturation, Input("S"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Value, Input("V"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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

  ICudaImage* hue = getHue().get();
  ICudaImage* saturation = getSaturation().get();
  ICudaImage* value = getValue().get();

  if (hue)
    hue->saveDeviceToWorkingCopy();
  if (saturation)
    saturation->saveDeviceToWorkingCopy();
  if (value)
    value->saveDeviceToWorkingCopy();

  int width = 0;
  int height = 0;
  int pixelWidth = 1000, pixelHeight = 1000;

  if (hue) {
    width = hue->getSize().x;
    height = hue->getSize().y;
    pixelWidth = hue->getVoxelSize().x;
    pixelHeight = hue->getVoxelSize().y;
  } else if (saturation) {
    width = saturation->getSize().x;
    height = saturation->getSize().y;
    pixelWidth = saturation->getVoxelSize().x;
    pixelHeight = saturation->getVoxelSize().y;
  } else if (value) {
    width = value->getSize().x;
    height = value->getSize().y;
    pixelWidth = value->getVoxelSize().x;
    pixelHeight = value->getVoxelSize().y;
  }

  if (width <= 0 || height <= 0)
    return;

  if (saturation) {
    if (saturation->getSize().x != width || saturation->getSize().y != height ||
        saturation->getVoxelSize().x != pixelWidth || saturation->getVoxelSize().y != pixelHeight)
      return;
  }
  if (value) {
    if (value->getSize().x != width || value->getSize().y != height ||
        value->getVoxelSize().x != pixelWidth || value->getVoxelSize().y != pixelHeight)
      return;
  }

  boost::shared_ptr<QImage> image(new QImage(width, height, QImage::Format_ARGB32));
  image->setDotsPerMeterX(1000000/pixelWidth);
  image->setDotsPerMeterY(1000000/pixelHeight);
  float* hueBuffer = (hue ? hue->getWorkingCopy() : 0);
  float* saturationBuffer = (saturation ? saturation->getWorkingCopy() : 0);
  float* valueBuffer = (value ? value->getWorkingCopy() : 0);

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
