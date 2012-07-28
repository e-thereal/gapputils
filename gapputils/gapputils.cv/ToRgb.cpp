#include "ToRgb.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ToRgb)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Input("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Red, Output("R"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Green, Output("G"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Blue, Output("B"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

ToRgb::ToRgb(void) : data(0)
{
  setLabel("ToRgb");
  Changed.connect(capputils::EventHandler<ToRgb>(this, &ToRgb::changedEventHandler));
}

ToRgb::~ToRgb(void)
{
  if (data)
    delete data;
}

void ToRgb::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

void ToRgb::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ToRgb();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getImagePtr())
    return;

  boost::shared_ptr<QImage> image = getImagePtr();
  const int width = image->width();
  const int height = image->height();
  boost::shared_ptr<image_t> redImage(new image_t(width, height, 1));
  boost::shared_ptr<image_t> greenImage(new image_t(width, height, 1));
  boost::shared_ptr<image_t> blueImage(new image_t(width, height, 1));
  float* redBuffer = redImage->getData();
  float* greenBuffer = greenImage->getData();
  float* blueBuffer = blueImage->getData();

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      QColor color(image->pixel(x, y));
      redBuffer[i] = (float)color.red() / 256.f;
      greenBuffer[i] = (float)color.green() / 256.f;
      blueBuffer[i] = (float)color.blue() / 256.f;
    }
  }

  data->setRed(redImage);
  data->setGreen(greenImage);
  data->setBlue(blueImage);
}

void ToRgb::writeResults() {
  if (!data)
    return;

  setRed(data->getRed());
  setGreen(data->getGreen());
  setBlue(data->getBlue());
}

}

}
