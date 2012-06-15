#pragma once
#ifndef _IMAGEVIEWER_H_
#define _IMAGEVIEWER_H_

#include <gapputils/WorkflowElement.h>

#include <string>
#include <qimage.h>

#include "ImageLoader.h"
#include "ShowImageDialog.h"

namespace gapputils {

class ImageViewer : public gapputils::workflow::WorkflowElement
{
private:
  InitReflectableClass(ImageViewer);

  Property(ImagePtr, boost::shared_ptr<QImage>)

private:
  ShowImageDialog* dialog;
  static int imageId;

public:
  ImageViewer(void);
  ~ImageViewer(void);

  virtual void execute(workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changeHandler(capputils::ObservableClass* sender, int eventId);

  virtual void show();
};

}

#endif
