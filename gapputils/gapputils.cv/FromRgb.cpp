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
#include <gapputils/ReadOnlyAttribute.h>

#include <QColor>
#include <cmath>

using namespace gapputils::attributes;
using namespace capputils::attributes;

namespace gapputils {

namespace cv {

int FromRgb::outputId;

BeginPropertyDefinitions(FromRgb)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Output("Img"), Volatile(), ReadOnly(), Observe(outputId = Id), TimeStamp(Id))
  DefineProperty(Red, Input("R"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Green, Input("G"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Blue, Input("B"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

FromRgb::FromRgb(void) : data(0)
{
  setLabel("FromRgb");
  Changed.connect(capputils::EventHandler<FromRgb>(this, &FromRgb::changedEventHandler));
}

FromRgb::~FromRgb(void)
{
  if (data)
    delete data;
}

void FromRgb::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId != outputId) {
    execute(0);
    writeResults();
  }
}

int toIntC(float value) {
  return std::min(255, std::max(0, (int)(value * 256)));
}

void FromRgb::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FromRgb();

  if (!capputils::Verifier::Valid(*this))
    return;

  image_t* red = getRed().get();
  image_t* green = getGreen().get();
  image_t* blue = getBlue().get();

  int width = 0;
  int height = 0;

  if (red) {
    width = red->getSize()[0];
    height = red->getSize()[1];
  } else if (green) {
    width = green->getSize()[0];
    height = green->getSize()[1];
  } else if (blue) {
    width = blue->getSize()[0];
    height = blue->getSize()[1];
  }

  if (width <= 0 || height <= 0)
    return;

  if (green) {
    if (green->getSize()[0] != width || green->getSize()[1] != height)
      return;
  }
  if (blue) {
    if (blue->getSize()[0] != width || blue->getSize()[1] != height)
      return;
  }

  
  boost::shared_ptr<QImage> image(new QImage(width, height, QImage::Format_ARGB32));
  float* redBuffer = (red ? red->getData() : 0);
  float* greenBuffer = (green ? green->getData() : 0);
  float* blueBuffer = (blue ? blue->getData() : 0);

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      int r = (redBuffer ? toIntC(redBuffer[i]) : 0);
      int g = (greenBuffer ? toIntC(greenBuffer[i]) : 0);
      int b = (blueBuffer ? toIntC(blueBuffer[i]) : 0);
      image->setPixel(x, y, QColor(r, g, b).rgb());
    }
  }

  data->setImagePtr(image);
}

void FromRgb::writeResults() {
  if (!data)
    return;

  setImagePtr(data->getImagePtr());
}

}

}
