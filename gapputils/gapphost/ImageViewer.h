#pragma once
#ifndef _IMAGEVIEWER_H_
#define _IMAGEVIEWER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

#include <string>
#include <qimage.h>

#include "ImageLoader.h"
#include "ShowImageDialog.h"

namespace gapputils {

class ImageViewer : public capputils::reflection::ReflectableClass, public capputils::ObservableClass
{
private:

  class ChangeEventHandler {
  private:
    ImageViewer* viewer;
  public:
    ChangeEventHandler(ImageViewer* viewer) : viewer(viewer) { }

    void operator()(capputils::ObservableClass* object, int eventId) {
      switch (eventId) {
      case 0:
        viewer->dialog->setWindowTitle(QString("Image Viewer: ") + viewer->getLabel().c_str());
        break;
      case 1:
        QImage* image = viewer->getImagePtr();
        if (image) {
          viewer->dialog->setImage(image);
        }
        break;
      }
    }
  } changeHandler;

  InitReflectableClass(ImageViewer);

  Property(Label, std::string)
  Property(ImagePtr, QImage*)

private:
  ShowImageDialog* dialog;

public:
  ImageViewer(void);
  ~ImageViewer(void);

  void showImage();
};

}

#endif
