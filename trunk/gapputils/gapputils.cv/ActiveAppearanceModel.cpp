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

  DefineProperty(MeanGrid, Volatile())
  DefineProperty(MeanImage, Volatile())
  DefineProperty(PrincipalGrids, Volatile())
  DefineProperty(PrincipalImages, Volatile())
  DefineProperty(PrincipalParameters, Volatile())
  DefineProperty(RowCount)
  DefineProperty(ColumnCount)
  DefineProperty(Width)
  DefineProperty(Height)

EndPropertyDefinitions

ActiveAppearanceModel::ActiveAppearanceModel()
 : _MeanGrid((std::vector<float>*)0),
   _MeanImage((std::vector<float>*)0),
   _PrincipalGrids((std::vector<float>*)0),
   _PrincipalImages((std::vector<float>*)0),
   _PrincipalParameters((std::vector<float>*)0)
{
}

ActiveAppearanceModel::~ActiveAppearanceModel() {
}

boost::shared_ptr<GridModel> ActiveAppearanceModel::createMeanGrid() {
  boost::shared_ptr<GridModel> grid(new GridModel());
  grid->setRowCount(getRowCount());
  grid->setColumnCount(getColumnCount());

  vector<float>* features = getMeanGrid().get();
  assert(2 * grid->getPoints()->size() == features->size());
  for (unsigned i = 0; i < grid->getPoints()->size(); ++i) {
    grid->getPoints()->at(i)->setX(features->at(2*i));
    grid->getPoints()->at(i)->setY(features->at(2 * i + 1));
  }

  return grid;
}

boost::shared_ptr<culib::ICudaImage> ActiveAppearanceModel::createMeanImage() {
  boost::shared_ptr<culib::ICudaImage> image(new culib::CudaImage(dim3(getWidth(), getHeight())));

  float* buffer = image->getOriginalImage();
  copy(getMeanImage()->begin(), getMeanImage()->end(), buffer);
  image->resetWorkingCopy();

  return image;
}

void ActiveAppearanceModel::setMeanGrid(boost::shared_ptr<GridModel> grid) {
  setRowCount(grid->getRowCount());
  setColumnCount(grid->getColumnCount());

  setMeanGrid(toFeatures(grid));
}

void ActiveAppearanceModel::setMeanImage(boost::shared_ptr<culib::ICudaImage> image) {
  setWidth(image->getSize().x);
  setHeight(image->getSize().y);

  setMeanImage(toFeatures(image));
}

boost::shared_ptr<vector<float> > ActiveAppearanceModel::toFeatures(boost::shared_ptr<GridModel> grid) {
  boost::shared_ptr<vector<float> > features(new vector<float>());
  assert(grid->getColumnCount() * grid->getRowCount() == (int)grid->getPoints()->size());

  const unsigned count = grid->getPoints()->size();
  for (unsigned i = 0; i < count; ++i) {
    features->push_back(grid->getPoints()->at(i)->getX());
    features->push_back(grid->getPoints()->at(i)->getY());
  }

  return features;
}

boost::shared_ptr<vector<float> > ActiveAppearanceModel::toFeatures(boost::shared_ptr<culib::ICudaImage> image) {
  boost::shared_ptr<vector<float> > features(new vector<float>());
  unsigned width = image->getSize().x;
  unsigned height = image->getSize().y;

  const unsigned count = width * height;
  features->resize(count);

  float* buffer = image->getWorkingCopy();
  copy(buffer, buffer + count, features->begin());

  return features;
}

}

}
