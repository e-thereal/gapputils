#include "ImageLoader.h"

#include <capputils/OutputAttribute.h>
#include <gapputils/LabelAttribute.h>

#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageLoader)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(ImageName, Observe(PROPERTY_ID), Filename(), FileExists())
  DefineProperty(ImagePtr, Output("Img"), Observe(PROPERTY_ID), Hide(), Volatile())

EndPropertyDefinitions

ImageLoader::ImageLoader(void) : changeHandler(this), _Label("Image"), _ImagePtr(0), image(0)
{
  Changed.connect(changeHandler);
}


ImageLoader::~ImageLoader(void)
{
  if (image)
    delete image;
}

void ImageLoader::loadImage() {
  if (!image)
    image = new QImage();
  if (image->load(getImageName().c_str()))
    setImagePtr(image);
}

}
