#include "ImageLoader.h"

#include <OutputAttribute.h>
#include <LabelAttribute.h>

#include <FilenameAttribute.h>
#include <ObserveAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageLoader)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(ImageName, Observe(PROPERTY_ID), Filename())
  DefineProperty(ImagePtr, Output(), Observe(PROPERTY_ID))

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
  if (image->load(getImageName().c_str()));
    setImagePtr(image);
}

}
