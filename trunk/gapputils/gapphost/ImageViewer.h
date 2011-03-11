#pragma once
#ifndef _IMAGEVIEWER_H_
#define _IMAGEVIEWER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

#include <string>
#include <qimage.h>

#include "ImageLoader.h"

namespace gapputils {

class ImageViewer : public capputils::reflection::ReflectableClass, public capputils::ObservableClass
{
  InitReflectableClass(ImageViewer);

  Property(Label, std::string)
  Property(ImagePtr, QImage*)

public:
  ImageViewer(void);
  ~ImageViewer(void);
};

}

#endif
