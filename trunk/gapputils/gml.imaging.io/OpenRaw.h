#pragma once
#ifndef GML_OPENRAW_H_
#define GML_OPENRAW_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace io {

CapputilsEnumerator(WhiteBalance, None, Auto, Camera);
CapputilsEnumerator(ColorSpace, Raw, sRGB, AdobeRGB, WideRGB, ProPhotoRGB, XYZ);
CapputilsEnumerator(InterpolationQuality, Linear, VNG, PPG, AHD, DCB, ModifiedAHD, AFD, VCD, MixedVCDModifiedAHD, LMMSE, AMaZE);

class OpenRaw : public DefaultWorkflowElement<OpenRaw> {

  InitReflectableClass(OpenRaw);

  Property(Filename, std::string)
  Property(HalfSize, bool)
  Property(WhiteBalance, WhiteBalance)
  Property(ColorSpace, ColorSpace)
  Property(InterpolationQuality, InterpolationQuality)
  Property(Image, boost::shared_ptr<image_t>)

public:
  OpenRaw(void);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif
