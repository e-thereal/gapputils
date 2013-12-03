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
#include <tbblas/tensor.hpp>

#include "ImageViewerDialog.h"

namespace gml {

namespace imaging {

namespace ui {

CapputilsEnumerator(ViewMode, Greyscale, RedBlueMap, HeatMap1, HeatMap2, sRGB, XYZ, xyY, CIELAB);

class ImageViewerWidget;

class ImageViewer : public DefaultWorkflowElement<ImageViewer> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  friend class ImageViewerWidget;

  InitReflectableClass(ImageViewer)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Tensors, boost::shared_ptr<v_tensor_t>)
  Property(AutoUpdateCurrentModule, bool)
  Property(AutoUpdateWorkflow, bool)
  Property(CurrentImage, int)
  Property(CurrentSlice, int)
  Property(MinimumIntensity, double)
  Property(MaximumIntensity, double)
  Property(Contrast, double)
  Property(Mode, ViewMode)
  Property(MinimumLength, double)
  Property(MaximumLength, double)
  Property(VisibleLength, double)
  //Property(WobbleDelay, int)

public:
  static int imageId, imagesId, tensorId, tensorsId, modeId, currentImageId, currentSliceId, minimumIntensityId, maximumIntensityId, minimumLengthId, maximumLengthId;

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
