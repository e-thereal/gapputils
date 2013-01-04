/*
 * ImageViewer.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GML_IMAGING_UI_IMAGEVIEWER_H_
#define GML_IMAGING_UI_IMAGEVIEWER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "ImageViewerDialog.h"

namespace gml {

namespace imaging {

namespace ui {

class ImageViewer : public DefaultWorkflowElement<ImageViewer> {

  InitReflectableClass(ImageViewer)

  Property(BackgroundImage, boost::shared_ptr<image_t>)

private:
  boost::shared_ptr<ImageViewerDialog> dialog;
  static int backgroundId;

public:
  ImageViewer();
  virtual ~ImageViewer();

  virtual void show();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

}

#endif /* GAPPUTILSCV_IMAGEVIEWER_H_ */
