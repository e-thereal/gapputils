/*
 * AamBuilder.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "AamBuilder.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include "ImageWarp.h"

#include <culib/CudaImage.h>
#include <culib/pca.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamBuilder)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Grids, Input(), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Images, Input(), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ActiveAppearanceModel, Output("AAM"), Hide(), Volatile(), Reflectable<boost::shared_ptr<ActiveAppearanceModel> >(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

//  DefineProperty(MeanGrid, Output("MG"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MeanImage, Output("MI"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamBuilder::AamBuilder() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamBuilder");

  Changed.connect(capputils::EventHandler<AamBuilder>(this, &AamBuilder::changedHandler));
}

AamBuilder::~AamBuilder() {
  if (data)
    delete data;
}

void AamBuilder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamBuilder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamBuilder();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<ActiveAppearanceModel> model(new ActiveAppearanceModel());
  data->setActiveAppearanceModel(model);

  calculateGrids();
  calculateImages();
}

void AamBuilder::writeResults() {
  if (!data)
    return;

  setActiveAppearanceModel(data->getActiveAppearanceModel());
//  setMeanGrid(getActiveAppearanceModel()->getMeanGrid());
  setMeanImage(getActiveAppearanceModel()->createMeanImage());
}

void AamBuilder::calculateGrids() const {
  boost::shared_ptr<ActiveAppearanceModel> model = data->getActiveAppearanceModel();

  std::vector<boost::shared_ptr<GridModel> >* grids = getGrids().get();
  if (!grids || grids->size() == 0)
    return;

  const int rowCount = grids->at(0)->getRowCount();
  const int columnCount = grids->at(0)->getColumnCount();
  const int count = rowCount * columnCount;
  const int gridCount = grids->size();

  for (unsigned i = 1; i < grids->size(); ++i) {
    if (grids->at(i)->getRowCount() != rowCount ||
        grids->at(i)->getColumnCount() != columnCount)
    {
      return;
    }
  }

  boost::shared_ptr<GridModel> meanGrid(new GridModel());
  meanGrid->setRowCount(rowCount);
  meanGrid->setColumnCount(columnCount);

  std::vector<GridPoint*>* meanPoints = meanGrid->getPoints();
  for (unsigned i = 0; i < meanPoints->size(); ++i) {
    float meanX = 0.0, meanY = 0.0;
    for (unsigned j = 0; j < grids->size(); ++j) {
      meanX += grids->at(j)->getPoints()->at(i)->getX();
      meanY += grids->at(j)->getPoints()->at(i)->getY();
    }
    meanPoints->at(i)->setX(meanX / grids->size());
    meanPoints->at(i)->setY(meanY / grids->size());
  }

  model->setMeanGrid(meanGrid);

  // Flatten grid points into one large vector
  // Put all vectors together into one large matrix
  // Subtract the mean grid from each grid

  vector<float>* meanFeatures = model->getMeanGrid().get();
  vector<float> featureMatrix(2 * count * gridCount);
  for (int i = 0, k = 0; i < gridCount; ++i) {
    boost::shared_ptr<std::vector<float> > features = ActiveAppearanceModel::toFeatures(grids->at(i));
    for (int j = 0; j < 2 * count; ++j, ++k) {
      featureMatrix[k] = features->at(j) - meanFeatures->at(j);
    }
  }

  // Calculate principal components of that matrix

  boost::shared_ptr<vector<float> > principalGrids(new vector<float> (4 * count * count));
  culib::getPcs(&(*principalGrids)[0], &featureMatrix[0], 2 * count, gridCount);

  // The result are the principal grids.
  model->setPrincipalGrids(principalGrids);
}

#define TRACE std::cout << __LINE__ << std::endl;

void AamBuilder::calculateImages() const {
  if (getImages()->size() != getGrids()->size())
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = data->getActiveAppearanceModel();

  if (!model)
    return;

  if (getImages()->size() == 0)
    return;

  const unsigned width = getImages()->at(0)->getSize().x;
  const unsigned height = getImages()->at(0)->getSize().y;
  const int count = width * height;
  const int imageCount = getImages()->size();

  // Warp all images into reference frame and calculate mean image afterwards
  std::vector<boost::shared_ptr<culib::ICudaImage> > warpedImages;
  boost::shared_ptr<GridModel> meanGrid = model->createMeanGrid();

  ImageWarp warp;
  for (int i = 0; i < imageCount; ++i) {
    if (width != getImages()->at(i)->getSize().x ||
        height != getImages()->at(i)->getSize().y)
    {
      continue;
    }
    warp.setBaseGrid(getGrids()->at(i));

    warp.setWarpedGrid(meanGrid);
    warp.setInputImage(getImages()->at(i));
    warp.execute(0);
    warp.writeResults();
    warpedImages.push_back(warp.getOutputImage());
  }

  boost::shared_ptr<culib::ICudaImage> meanImage(new culib::CudaImage(dim3(width, height)));

  float* buffer = meanImage->getOriginalImage();
  for (unsigned i = 0; i < width * height; ++i) {
    float value = 0;
    for (unsigned j = 0; j < warpedImages.size(); ++j) {
      value += warpedImages[j]->getWorkingCopy()[i];
    }
    buffer[i] = value / warpedImages.size();
  }
  meanImage->resetWorkingCopy();
  model->setMeanImage(meanImage);

  return;

  // Flatten pixels into one large vector
  // Put all vectors together into one large matrix
  // Subtract the mean image from each image

  vector<float>* meanFeatures = model->getMeanImage().get();
  vector<float> featureMatrix(count * imageCount);
  for (int i = 0, k = 0; i < imageCount; ++i) {
    boost::shared_ptr<std::vector<float> > features = ActiveAppearanceModel::toFeatures(warpedImages.at(i));
    for (int j = 0; j < count; ++j, ++k) {
      featureMatrix[k] = features->at(j) - meanFeatures->at(j);
    }
  }

  // Calculate principal components of that matrix

  boost::shared_ptr<vector<float> > principalImages(new vector<float> (count * count));
  culib::getPcs(&(*principalImages)[0], &featureMatrix[0], count, imageCount);

  // The result are the principal images.
  model->setPrincipalImages(principalImages);
}

}

}
