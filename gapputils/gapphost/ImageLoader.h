#pragma once
#ifndef _IMAGELOADER_H_
#define _IMAGELOADER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>
#include <string>

#include <qimage.h>

namespace capputils {
  namespace reflection {
    template<>
    const std::string convertToString(const QImage& image);
  }
}

namespace gapputils {

class ImageLoader : public capputils::reflection::ReflectableClass, public capputils::ObservableClass
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
};

}

#endif