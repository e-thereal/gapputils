#pragma once
#ifndef GML_CONVERTCOLORSPACE_H_
#define GML_CONVERTCOLORSPACE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(ColorSpace, sRGB, XYZ, xyY);

class ConvertColorSpace : public DefaultWorkflowElement<ConvertColorSpace> {

  InitReflectableClass(ConvertColorSpace)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(InputColorSpace, ColorSpace)
  Property(OutputColorSpace, ColorSpace)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  ConvertColorSpace();

protected:
  virtual void update(IProgressMonitor* monitor) const;

};

}

}

}

#endif