#pragma once
#ifndef GML_IMAGING_IO_IMAGEREADER_H
#define GML_IMAGING_IO_IMAGEREADER_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <gapputils/Image.h>

namespace gml {

namespace imaging {

namespace io {

class ImageReader : public DefaultWorkflowElement<ImageReader>
{
  InitReflectableClass(ImageReader)

  Property(ImageName, std::string)
  Property(ImagePtr, boost::shared_ptr<image_t>)
  Property(Width, int)
  Property(Height, int)

public:
  ImageReader(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_IMAGING_IO_IMAGEREADER_H */
