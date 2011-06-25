#include "FromRgb.h"


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

#include <QColor>
#include <cmath>

using namespace gapputils::attributes;

using namespace capputils::attributes;
using namespace culib;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(FromRgb)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Output("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Red, Input("R"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Green, Input("G"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Blue, Input("B"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FromRgb::FromRgb(void) : _ImagePtr(0), _Red(0), _Green(0), _Blue(0), data(0)
{
  setLabel("FromRgb");
  Changed.connect(capputils::EventHandler<FromRgb>(this, &FromRgb::changedEventHandler));
}

FromRgb::~FromRgb(void)
{
  if (data)
    delete data;

  if (_ImagePtr)
    delete _ImagePtr;
}

void FromRgb::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

int toIntC(float value) {
  return std::min(255, std::max(0, (int)(value * 256)));
}

void FromRgb::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FromRgb();

  if (!capputils::Verifier::Valid(*this))
    return;

  ICudaImage* red = getRed();
  ICudaImage* green = getGreen();
  ICudaImage* blue = getBlue();

  if (red)
    red->saveDeviceToWorkingCopy();
  if (green)
    green->saveDeviceToWorkingCopy();
  if (blue)
    blue->saveDeviceToWorkingCopy();

  int width = 0;
  int height = 0;

  if (red) {
    width = red->getSize().x;
    height = red->getSize().y;
  } else if (green) {
    width = green->getSize().x;
    height = green->getSize().y;
  } else if (blue) {
    width = blue->getSize().x;
    height = blue->getSize().y;
  }

  if (width <= 0 || height <= 0)
    return;

  if (green) {
    if (green->getSize().x != width || green->getSize().y != height)
      return;
  }
  if (blue) {
    if (blue->getSize().x != width || blue->getSize().y != height)
      return;
  }

  
  QImage* image = new QImage(width, height, QImage::Format_ARGB32);
  float* redBuffer = (red ? red->getWorkingCopy() : 0);
  float* greenBuffer = (green ? green->getWorkingCopy() : 0);
  float* blueBuffer = (blue ? blue->getWorkingCopy() : 0);

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      int r = (redBuffer ? toIntC(redBuffer[i]) : 0);
      int g = (greenBuffer ? toIntC(greenBuffer[i]) : 0);
      int b = (blueBuffer ? toIntC(blueBuffer[i]) : 0);
      image->setPixel(x, y, QColor(r, g, b).rgb());
    }
  }

  if (data->getImagePtr())
    delete data->getImagePtr();
  data->setImagePtr(image);
}

void FromRgb::writeResults() {
  if (!data)
    return;

  if (getImagePtr())
    delete getImagePtr();
  setImagePtr(data->getImagePtr());
  data->setImagePtr(0);
}

}

}
