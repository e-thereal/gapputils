#pragma once
#ifndef _IMAGELOADER_H_
#define _IMAGELOADER_H_

#include <string>

#include <qimage.h>
#include <WorkflowElement.h>

namespace capputils {

namespace reflection {

template<>
class Converter<QImage, false> {
public:
  static std::string toString(const QImage& value) {
    return std::string("[QImage]");
  }
};
  
}

}

namespace gapputils {

class ImageLoader : public gapputils::workflow::WorkflowElement
{
  class ChangeEventHandler {
  private:
    ImageLoader* loader;
  public:
    ChangeEventHandler(ImageLoader* loader) : loader(loader) { }

    void operator()(capputils::ObservableClass*, int eventId) {
      if (eventId == 1)
        loader->loadImage();
    }
  } changeHandler;

  InitReflectableClass(ImageLoader)

  Property(Label, std::string)
  Property(ImageName, std::string)
  Property(ImagePtr, QImage*)

private:
  QImage* image;

public:
  ImageLoader(void);
  virtual ~ImageLoader(void);

  void loadImage();
  virtual void execute(workflow::IProgressMonitor* monitor) const { }
  virtual void writeResults() { }

};

}

#endif
