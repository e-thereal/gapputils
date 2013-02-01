#include "ImageLoader.h"

#include <capputils/OutputAttribute.h>
#include <gapputils/LabelAttribute.h>

#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/NoParameterAttribute.h>

#include <iostream>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageLoader)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(ImageName, Input("Name"), Observe(Id), Filename(), FileExists(), TimeStamp(Id))
  DefineProperty(ImagePtr, Output("Img"), Observe(Id), ReadOnly(), Volatile(), TimeStamp(Id))
  DefineProperty(Width, Observe(Id), TimeStamp(Id), NoParameter())
  DefineProperty(Height, Observe(Id), TimeStamp(Id), NoParameter())

EndPropertyDefinitions

ImageLoader::ImageLoader(void) : _Width(0), _Height(0), data(0)
{
  setLabel("Image");
}

ImageLoader::~ImageLoader(void)
{
  if (data)
    delete data;
}

void ImageLoader::execute(workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageLoader();

  boost::shared_ptr<QImage> image(new QImage());
  if (!image->load(getImageName().c_str())) {
    std::cout << "[Warning] Could not load image " << getImageName() << std::endl;
    return;
  }

  data->setWidth(image->width());
  data->setHeight(image->height());
  data->setImagePtr(image);
}

void ImageLoader::writeResults() {
  if (!data)
    return;

  setImagePtr(data->getImagePtr());
  setWidth(data->getWidth());
  setHeight(data->getHeight());
}

}
