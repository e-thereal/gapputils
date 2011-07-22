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

#include <iostream>

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
  DefineProperty(Map, Output(), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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

float3 toBarycentric(const float2& p, const float2& p1, const float2& p2, const float2& p3) {
  /*if (p.x == p1.x && p.y == p1.y)
    return make_float3(1, 0, 0);
  else if (p.x == p2.x && p.y == p2.y)
    return make_float3(0, 1, 0);
  else if (p.x == p3.x && p.y == p3.y)
    return make_float3(0, 0, 1);
  */

  float a = ((p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y)) /
    ((p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y));
  float b = ((p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y  - p3.y)) /
    ((p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y));
  float c = 1 - a - b;

  return make_float3(a, b, c);
}

float2 toCartesian(const float3& p, const float2& p1, const float2& p2, const float2& p3) {
  return make_float2(p.x * p1.x + p.y * p2.x + p.z * p3.x, p.x * p1.y + p.y * p2.y + p.z * p3.y);
}

bool withinTriangle(const float2& p, const float2& p1, const float2& p2, const float2& p3) {
  float3 barys = toBarycentric(p, p1, p2, p3);

  return barys.x >= 0 && barys.y >= 0 && barys.z >= 0;
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

  std::vector<GridPoint*>* baseGridPoints = baseGrid->getPoints();
  std::vector<GridPoint*>* warpedGridPoints = warpedGrid->getPoints();

  if (warpedGrid->getRowCount() != rowCount || warpedGrid->getColumnCount() != columnCount)
    return;

  ICudaImage* inputImage = getInputImage().get();
  boost::shared_ptr<ICudaImage> bgImage = getBackgroundImage();
  const int width = inputImage->getSize().x;
  const int height = inputImage->getSize().y;

  boost::shared_ptr<ICudaImage> warpedImage(new CudaImage(inputImage->getSize(), inputImage->getVoxelSize()));
  inputImage->saveDeviceToWorkingCopy();
  if (bgImage) bgImage->saveDeviceToWorkingCopy();
  float* inputBuffer = inputImage->getWorkingCopy();
  float* warpedBuffer = warpedImage->getWorkingCopy();
  float* bgBuffer = (bgImage ? bgImage->getWorkingCopy() : 0);

  // Build the map first
  std::vector<float2> wp0s, wp1s, wp2s, bp0s, bp1s, bp2s;
  boost::shared_ptr<QImage> triangleMap(new QImage(width, height, QImage::Format_ARGB32));
  QPainter painter(triangleMap.get());
  painter.fillRect(0, 0, width, height, Qt::black);

  for (int iRow = 1, i = 0; iRow < rowCount; ++iRow) {
    for (int iCol = 1; iCol < columnCount; ++iCol) {
      // check all 4 triangles
      GridPoint* wgp1 = warpedGridPoints->at(iCol + iRow * columnCount);
      GridPoint* wgp2 = warpedGridPoints->at(iCol + (iRow - 1) * columnCount);
      GridPoint* wgp3 = warpedGridPoints->at((iCol - 1) + (iRow - 1) * columnCount);
      GridPoint* wgp4 = warpedGridPoints->at((iCol - 1) + iRow * columnCount);

      float2 wp1 = make_float2(wgp1->getX(), wgp1->getY());
      float2 wp2 = make_float2(wgp2->getX(), wgp2->getY());
      float2 wp3 = make_float2(wgp3->getX(), wgp3->getY());
      float2 wp4 = make_float2(wgp4->getX(), wgp4->getY());
      float2 wmid = 0.25 * wp1 + 0.25 * wp2 + 0.25 * wp3 + 0.25 * wp4;

      GridPoint* bgp1 = baseGridPoints->at(iCol + iRow * columnCount);
      GridPoint* bgp2 = baseGridPoints->at(iCol + (iRow - 1) * columnCount);
      GridPoint* bgp3 = baseGridPoints->at((iCol - 1) + (iRow - 1) * columnCount);
      GridPoint* bgp4 = baseGridPoints->at((iCol - 1) + iRow * columnCount);

      float2 bp1 = make_float2(bgp1->getX(), bgp1->getY());
      float2 bp2 = make_float2(bgp2->getX(), bgp2->getY());
      float2 bp3 = make_float2(bgp3->getX(), bgp3->getY());
      float2 bp4 = make_float2(bgp4->getX(), bgp4->getY());
      float2 bmid = 0.25 * bp1 + 0.25 * bp2 + 0.25 * bp3 + 0.25 * bp4;

      { // upper triangle
        QPainterPath path;
        QPolygonF polygon;
        polygon.append(QPointF(wp2.x, wp2.y));
        polygon.append(QPointF(wp3.x, wp3.y));
        polygon.append(QPointF(wmid.x, wmid.y));
        path.addPolygon(polygon);
        painter.fillPath(path, QBrush(QColor(QRgb(++i))));
        wp0s.push_back(wp2);
        wp1s.push_back(wp3);
        wp2s.push_back(wmid);
        bp0s.push_back(bp2);
        bp1s.push_back(bp3);
        bp2s.push_back(bmid);
      }

      { // left triangle
        QPainterPath path;
        QPolygonF polygon;
        polygon.append(QPointF(wp3.x, wp3.y));
        polygon.append(QPointF(wp4.x, wp4.y));
        polygon.append(QPointF(wmid.x, wmid.y));
        path.addPolygon(polygon);
        painter.fillPath(path, QBrush(QColor(QRgb(++i))));
        wp0s.push_back(wp3);
        wp1s.push_back(wp4);
        wp2s.push_back(wmid);
        bp0s.push_back(bp3);
        bp1s.push_back(bp4);
        bp2s.push_back(bmid);
      }

      { // lower triangle
        QPainterPath path;
        QPolygonF polygon;
        polygon.append(QPointF(wp4.x, wp4.y));
        polygon.append(QPointF(wp1.x, wp1.y));
        polygon.append(QPointF(wmid.x, wmid.y));
        path.addPolygon(polygon);
        painter.fillPath(path, QBrush(QColor(QRgb(++i))));
        wp0s.push_back(wp4);
        wp1s.push_back(wp1);
        wp2s.push_back(wmid);
        bp0s.push_back(bp4);
        bp1s.push_back(bp1);
        bp2s.push_back(bmid);
      }

      { // right triangle
        QPainterPath path;
        QPolygonF polygon;
        polygon.append(QPointF(wp1.x, wp1.y));
        polygon.append(QPointF(wp2.x, wp2.y));
        polygon.append(QPointF(wmid.x, wmid.y));
        path.addPolygon(polygon);
        painter.fillPath(path, QBrush(QColor(QRgb(++i))));
        wp0s.push_back(wp1);
        wp1s.push_back(wp2);
        wp2s.push_back(wmid);
        bp0s.push_back(bp1);
        bp1s.push_back(bp2);
        bp2s.push_back(bmid);
      }
    }
  }

  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {

      float2 p = make_float2((float)x + 0.5f, (float)y + 0.5f);
      float2 inP;
      bool foundTriangle = false;

      QColor color = triangleMap->pixel(x, y);
      unsigned index = color.rgb() & 0xFFFFFF;
      if (index && index <= wp0s.size()) {
        float3 barys = toBarycentric(p, wp0s[index - 1], wp1s[index - 1], wp2s[index - 1]);
        inP = toCartesian(barys, bp0s[index - 1], bp1s[index - 1], bp2s[index - 1]);
        foundTriangle = true;
      }
      
      if (!foundTriangle) {
        //std::cout << "No triangle found for (" << x << ", " << y << "). Index = " << index << std::endl;
        warpedBuffer[i] = (bgBuffer ? bgBuffer[i] : 0);
      } else {
        int inX = inP.x;
        int inY = inP.y;
        warpedBuffer[i] = inputBuffer[inX + inY * width];
      }
    }
    if (monitor)
      monitor->reportProgress(y * 100 / height);
  }

  data->setMap(triangleMap);
  data->setOutputImage(warpedImage);
}

void ImageWarp::writeResults() {
  if (!data)
    return;
  setOutputImage(data->getOutputImage());
  setMap(data->getMap());
}

}

}
