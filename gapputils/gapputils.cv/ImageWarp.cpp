#include "ImageWarp.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <qimage.h>
#include <qpainter.h>
#include <qpainterpath.h>
#include <qcolor.h>

#include <culib/CudaImage.h>
#include <culib/math3d.h>
#include <culib/transform.h>

#include <algorithm>
#include <iostream>
#include <cmath>

#include "ActiveAppearanceModel.h"
#include "cuda_util.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageWarp)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(InputImage, Input("Img"), Hide(), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BackgroundImage, Input("BG"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BaseGrid, Input("Base"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(WarpedGrid, Input("Warped"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ImageWarp::ImageWarp(void) : data(0)
{
  setLabel("ImageWarp");
  Changed.connect(capputils::EventHandler<ImageWarp>(this, &ImageWarp::changedEventHandler));
}

ImageWarp::~ImageWarp(void)
{
  if (data)
    delete data;
}

void ImageWarp::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

void ImageWarp::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace culib;

  if (!data)
    data = new ImageWarp();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage() || !getBaseGrid() || !getWarpedGrid())
    return;

  // For every point in the target frame get point in reference frame
  // - find triangle were points lies within
  // - calculate barycentric coordinates with respect to that triangle
  // - calculate Cartesian coordinates with respect to reference triangle
  // - nearest neighbor interpolation
  // 

  boost::shared_ptr<GridModel> baseGrid = getBaseGrid();
  boost::shared_ptr<GridModel> warpedGrid = getWarpedGrid();

  const int rowCount = baseGrid->getRowCount();
  const int columnCount = baseGrid->getColumnCount();

  if (warpedGrid->getRowCount() != rowCount || warpedGrid->getColumnCount() != columnCount)
    return;

  boost::shared_ptr<ICudaImage> inputImage = make_cuda_image(*getInputImage());
  boost::shared_ptr<ICudaImage> bgImage = make_cuda_image(*getBackgroundImage());

  boost::shared_ptr<ICudaImage> warpedImage((bgImage ? new CudaImage(*bgImage) : new CudaImage(inputImage->getSize(), inputImage->getVoxelSize())));
  warpImage(warpedImage->getDevicePointer(), inputImage->getCudaArray(), inputImage->getSize(),
      (float2*)baseGrid->getDeviceFeatures(), (float2*)warpedGrid->getDeviceFeatures(),
      dim3(columnCount, rowCount));

  warpedImage->saveDeviceToWorkingCopy();
  warpedImage->freeCaches();
  inputImage->freeCaches();
  baseGrid->freeCaches();
  warpedGrid->freeCaches();

  data->setOutputImage(make_gapputils_image(*warpedImage));
}

void ImageWarp::writeResults() {
  if (!data)
    return;
  setOutputImage(data->getOutputImage());
}

}

}
