#pragma once
#ifndef _IMAGELOADER_H_
#define _IMAGELOADER_H_

#include <string>

#include <qimage.h>
#include <gapputils/WorkflowElement.h>
#include <boost/shared_ptr.hpp>

namespace capputils {

namespace reflection {

template<>
class Converter<QImage, false> {
public:
  static std::string toString(const QImage& /*value*/) {
    return std::string("[QImage]");
  }
};
  
}

}

namespace gapputils {

class ImageLoader : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(ImageLoader)

  Property(ImageName, std::string)
  Property(ImagePtr, boost::shared_ptr<QImage>)
  Property(Width, int)
  Property(Height, int)

private:
  mutable ImageLoader* data;

public:
  ImageLoader(void);
  virtual ~ImageLoader(void);

  virtual void execute(workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
