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

#include "util.h"

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

boost::shared_ptr<image_t> ActiveAppearanceModel::createMeanTexture() {
  return createTexture(getMeanTexture().get());
}

boost::shared_ptr<image_t> ActiveAppearanceModel::createTexture(std::vector<float>* features) {
  if (!features) {
    return boost::shared_ptr<image_t>();
  }
  boost::shared_ptr<image_t> image(new image_t(getWidth(), getHeight(), 1));

  float* buffer = image->getData();
  copy(features->begin(), features->end(), buffer);

  return image;
}

void ActiveAppearanceModel::setMeanShape(boost::shared_ptr<GridModel> grid) {
  setRowCount(grid->getRowCount());
  setColumnCount(grid->getColumnCount());

  setMeanShape(toFeatures(grid.get()));
}

void ActiveAppearanceModel::setMeanTexture(boost::shared_ptr<image_t> image) {
  setWidth(image->getSize()[0]);
  setHeight(image->getSize()[1]);

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

boost::shared_ptr<vector<float> > ActiveAppearanceModel::toFeatures(image_t* image) {
  boost::shared_ptr<vector<float> > features(new vector<float>());
  unsigned width = image->getSize()[0];
  unsigned height = image->getSize()[1];

  const unsigned count = width * height;
  features->resize(count);

  float* buffer = image->getData();
  copy(buffer, buffer + count, features->begin());

  return features;
}

}

}
