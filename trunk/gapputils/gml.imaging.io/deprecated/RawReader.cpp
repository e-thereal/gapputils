#include "RawReader.h"

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(RawReader, Renamed("gml::imaging::io::OpenRaw"), Deprecated("Use OpenRaw instead."))

  ReflectableBase(DefaultWorkflowElement<RawReader>)

  WorkflowProperty(Filename, Input("Name"), Filename(), FileExists())
  WorkflowProperty(HalfSize, Description("Load the image at half of the original resolution."))
  WorkflowProperty(WhiteBalance, Enumerator<Type>())
  WorkflowProperty(ColorSpace, Enumerator<Type>())
  WorkflowProperty(InterpolationQuality, Enumerator<Type>())
  WorkflowProperty(Image, Output("Img"))

EndPropertyDefinitions

}

}

}

#include <libraw/libraw.h>

namespace gml {

namespace imaging {

namespace io {

RawReader::RawReader(void) : _HalfSize(false), _ColorSpace(ColorSpace::sRGB) { 
  setLabel("Raw");
}

void RawReader::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  LibRaw processor;
  processor.open_file(getFilename().c_str());
  processor.unpack();
  processor.imgdata.params.output_bps = 16;
  processor.imgdata.params.half_size = getHalfSize();
  switch(getWhiteBalance()) {
  case WhiteBalance::None:
    break;

  case WhiteBalance::Auto:
    processor.imgdata.params.use_auto_wb = 1;
    break;

  case WhiteBalance::Camera:
    processor.imgdata.params.use_camera_wb = 1;
    break;
  }
  processor.imgdata.params.output_color = getColorSpace();
  processor.imgdata.params.user_qual = getInterpolationQuality();
  processor.dcraw_process();

  auto image = processor.dcraw_make_mem_image();

  boost::shared_ptr<image_t> output(new image_t(image->width, image->height, 3));
  const size_t count = image->width * image->height;
  float* buffer = output->getData();
  unsigned short* data = (unsigned short*)image->data;

  for (size_t i = 0; i < count; ++i) {
    for (size_t iCol = 0; iCol < image->colors; ++iCol) {
      buffer[i + iCol * count] = (float)data[image->colors * i + iCol] / 65536.f;
    }
  }
  LibRaw::dcraw_clear_mem(image);
  processor.recycle();

  newState->setImage(output);
}

}

}

}
