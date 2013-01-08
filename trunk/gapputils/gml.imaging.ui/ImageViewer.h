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

#include <capputils/Enumerators.h>

#include "ImageViewerDialog.h"

namespace gml {

namespace imaging {

namespace ui {

CapputilsEnumerator(ViewMode, Default, Wobble);

class ImageViewer : public DefaultWorkflowElement<ImageViewer> {

  InitReflectableClass(ImageViewer)

  Property(BackgroundImage, boost::shared_ptr<image_t>)
  Property(Mode, ViewMode)
  Property(WobbleDelay, int)

public:
  static int backgroundId, modeId;

private:
  boost::shared_ptr<ImageViewerDialog> dialog;

public:
  ImageViewer();
  virtual ~ImageViewer();

  virtual void show();
};

}

}

}

#endif /* GAPPUTILSCV_IMAGEVIEWER_H_ */
