#include "ImageViewer.h"

#include "LabelAttribute.h"
#include "InputAttribute.h"
#include <ObserveAttribute.h>

using namespace capputils::attributes;


namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageViewer)

DefineProperty(Label, Label(), Observe(PROPERTY_ID))
DefineProperty(ImagePtr, Input(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ImageViewer::ImageViewer(void) : _Label("Viewer"), _ImagePtr(0)
{
}


ImageViewer::~ImageViewer(void)
{
}

}
