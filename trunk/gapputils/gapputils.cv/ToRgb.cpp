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

#include <culib/CudaImage.h>

#include <gapputils/HideAttribute.h>

#include <QColor>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ToRgb)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Input("Img"), Volatile(), Hide(), NotEqual<QImage*>(0), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Red, Output("R"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Green, Output("G"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Blue, Output("B"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ToRgb::ToRgb(void) : _ImagePtr(0), _Red(0), _Green(0), _Blue(0), data(0)
{
  setLabel("ToRgb");
  Changed.connect(capputils::EventHandler<ToRgb>(this, &ToRgb::changedEventHandler));
}

ToRgb::~ToRgb(void)
{
  if (data)
    delete data;

  if (_Red)
    delete _Red;
  if (_Green)
    delete _Green;
  if(_Blue)
    _Blue;
}

void ToRgb::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

void ToRgb::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace culib;

  if (!data)
    data = new ToRgb();

  if (!capputils::Verifier::Valid(*this))
    return;

  QImage* image = getImagePtr();
  const int width = image->width();
  const int height = image->height();
 
  CudaImage* redImage = new CudaImage(dim3(width, height));
  CudaImage* greenImage = new CudaImage(dim3(width, height));
  CudaImage* blueImage = new CudaImage(dim3(width, height));
  float* redBuffer = redImage->getOriginalImage();
  float* greenBuffer = greenImage->getOriginalImage();
  float* blueBuffer = blueImage->getOriginalImage();

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      QColor color(image->pixel(x, y));
      redBuffer[i] = (float)color.red() / 256.f;
      greenBuffer[i] = (float)color.green() / 256.f;
      blueBuffer[i] = (float)color.blue() / 256.f;
    }
  }
  redImage->resetWorkingCopy();
  greenImage->resetWorkingCopy();
  blueImage->resetWorkingCopy();

  if (data->getRed())
    delete data->getRed();
  data->setRed(redImage);

  if (data->getGreen())
    delete data->getGreen();
  data->setGreen(greenImage);

  if (data->getBlue())
    delete data->getBlue();
  data->setBlue(blueImage);
}

void ToRgb::writeResults() {
  if (!data)
    return;

  if (getRed())
    delete getRed();
  setRed(data->getRed());
  data->setRed(0);

  if (getGreen())
    delete getGreen();
  setGreen(data->getGreen());
  data->setGreen(0);

  if (getBlue())
    delete getBlue();
  setBlue(data->getBlue());
  data->setBlue(0);
}

}

}