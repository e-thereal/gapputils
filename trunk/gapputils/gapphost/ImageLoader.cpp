#include "ImageLoader.h"

#include "OutputAttribute.h"
#include "LabelAttribute.h"

#include <FilenameAttribute.h>
#include <ObserveAttribute.h>

namespace capputils {
  namespace reflection {
    template<>
    const std::string convertToString(const QImage& image) {
      return "[QImage]";
    }
  }
}

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageLoader)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(ImageName, Observe(PROPERTY_ID), Filename())
  DefineProperty(ImagePtr, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ImageLoader::ImageLoader(void) : _Label("Image"), _ImagePtr(0)
{
}


ImageLoader::~ImageLoader(void)
{
}

}
