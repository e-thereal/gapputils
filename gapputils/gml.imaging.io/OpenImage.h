#pragma once
#ifndef GML_IMAGING_IO_OPENIMAGE_H
#define GML_IMAGING_IO_OPENIMAGE_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <gapputils/Image.h>

namespace gml {

namespace imaging {

namespace io {

class OpenImage : public DefaultWorkflowElement<OpenImage>
{
  InitReflectableClass(OpenImage)

  Property(ImageName, std::string)
  Property(ImagePtr, boost::shared_ptr<image_t>)
  Property(Width, int)
  Property(Height, int)

public:
  OpenImage(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_IMAGING_IO_OPENIMAGE_H */
