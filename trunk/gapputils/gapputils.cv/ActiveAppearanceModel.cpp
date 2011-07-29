/*
 * ActiveAppearanceModel.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "ActiveAppearanceModel.h"

#include <capputils/VolatileAttribute.h>
#include <capputils/EnumerableAttribute.h>
#include <cassert>
#include <algorithm>
#include <iostream>

#include <culib/CudaImage.h>

#include <capputils/Xmlizer.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ActiveAppearanceModel)

  DefineProperty(MeanShape, Volatile())
  DefineProperty(MeanTexture, Volatile())
  DefineProperty(ShapeMatrix, Volatile())
  DefineProperty(TextureMatrix, Volatile())
  DefineProperty(AppearanceMatrix, Volatile())

  DefineProperty(SingularShapeParameters, Volatile())
  DefineProperty(SingularTextureParameters, Volatile())
  DefineProperty(SingularAppearanceParameters, Volatile())

  DefineProperty(RowCount)
  DefineProperty(ColumnCount)
  DefineProperty(Width)
  DefineProperty(Height)
  DefineProperty(ShapeParameterCount)
  DefineProperty(TextureParameterCount)
  DefineProperty(AppearanceParameterCount)

EndPropertyDefinitions

ActiveAppearanceModel::ActiveAppearanceModel() { }

ActiveAppearanceModel::~ActiveAppearanceModel() { }

boost::shared_ptr<GridModel> ActiveAppearanceModel::createMeanShape() {
  return createShape(getMeanShape().get());
}

boost::shared_ptr<GridModel> ActiveAppearanceModel::createShape(std::vector<float>* features) {
  boost::shared_ptr<GridModel> grid(new GridModel());
  grid->setRowCount(getRowCount());
  grid->setColumnCount(getColumnCount());
  
  assert(2 * grid->getPoints()->size() == features->size());
  for (unsigned i = 0; i < grid->getPoints()->size(); ++i) {
    grid->getPoints()->at(i)->setX(features->at(2*i));
    grid->getPoints()->at(i)->setY(features->at(2 * i + 1));
  }

  return grid;
}

boost::shared_ptr<culib::ICudaImage> ActiveAppearanceModel::createMeanTexture() {
  return createTexture(getMeanTexture().get());
}

boost::shared_ptr<culib::ICudaImage> ActiveAppearanceModel::createTexture(std::vector<float>* features) {
  if (!features) {
    boost::shared_ptr<culib::ICudaImage> image((culib::ICudaImage*)0);
    return image;
  }
  boost::shared_ptr<culib::ICudaImage> image(new culib::CudaImage(dim3(getWidth(), getHeight())));

  float* buffer = image->getOriginalImage();
  copy(features->begin(), features->end(), buffer);
  image->resetWorkingCopy();

  return image;
}

void ActiveAppearanceModel::setMeanShape(boost::shared_ptr<GridModel> grid) {
  setRowCount(grid->getRowCount());
  setColumnCount(grid->getColumnCount());

  setMeanShape(toFeatures(grid.get()));
}

void ActiveAppearanceModel::setMeanTexture(boost::shared_ptr<culib::ICudaImage> image) {
  setWidth(image->getSize().x);
  setHeight(image->getSize().y);

  setMeanTexture(toFeatures(image.get()));
}

boost::shared_ptr<vector<float> > ActiveAppearanceModel::toFeatures(GridModel* grid) {
  boost::shared_ptr<vector<float> > features(new vector<float>());
  assert(grid->getColumnCount() * grid->getRowCount() == (int)grid->getPoints()->size());

  const unsigned count = grid->getPoints()->size();
  for (unsigned i = 0; i < count; ++i) {
    features->push_back(grid->getPoints()->at(i)->getX());
    features->push_back(grid->getPoints()->at(i)->getY());
  }

  return features;
}

boost::shared_ptr<vector<float> > ActiveAppearanceModel::toFeatures(culib::ICudaImage* image) {
  boost::shared_ptr<vector<float> > features(new vector<float>());
  unsigned width = image->getSize().x;
  unsigned height = image->getSize().y;

  const unsigned count = width * height;
  features->resize(count);

  image->saveDeviceToWorkingCopy();
  float* buffer = image->getWorkingCopy();
  copy(buffer, buffer + count, features->begin());

  return features;
}

}

}
