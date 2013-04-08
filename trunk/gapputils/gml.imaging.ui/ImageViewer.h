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

CapputilsEnumerator(ViewMode, Greyscale, RedBlueMap, HeatMap1, HeatMap2, sRGB, XYZ, xyY, CIELAB);

class ImageViewerWidget;

class ImageViewer : public DefaultWorkflowElement<ImageViewer> {

  friend class ImageViewerWidget;

  InitReflectableClass(ImageViewer)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(CurrentImage, int)
  Property(CurrentSlice, int)
  Property(MinimumIntensity, double)
  Property(MaximumIntensity, double)
  Property(Contrast, double)
  Property(Mode, ViewMode)
  //Property(WobbleDelay, int)

public:
  static int imageId, imagesId, modeId, currentImageId, currentSliceId, minimumIntensityId, maximumIntensityId;

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
