/*
 * ImageViewer.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_IMAGEVIEWER_H_
#define GAPPUTILSCV_IMAGEVIEWER_H_

#include <gapputils/WorkflowElement.h>

#include <QImage>

#include "ImageViewerDialog.h"

namespace gapputils {

namespace cv {

class ImageViewer : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageViewer)

  Property(BackgroundImage, boost::shared_ptr<QImage>)

private:
  ImageViewerDialog* dialog;
  static int backgroundId;

public:
  ImageViewer();
  virtual ~ImageViewer();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
  virtual void show();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_IMAGEVIEWER_H_ */
