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
#include <culib/lintrans.h>
#include <cublas.h>

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
  DefineProperty(ShapeParameterCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(TextureParameterCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(AppearanceParameterCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

//  DefineProperty(MeanGrid, Output("MG"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MeanImage, Output("MI"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

#define TRACE std::cout << __LINE__ << std::endl;

AamBuilder::AamBuilder() : _ShapeParameterCount(10), _TextureParameterCount(20), _AppearanceParameterCount(10), data(0) {
  WfeUpdateTimestamp
  setLabel("AamBuilder");

  Changed.connect(capputils::EventHandler<AamBuilder>(this, &AamBuilder::changedHandler));
}

AamBuilder::~AamBuilder() {
  if (data)
    delete data;
}

void AamBuilder::changedHandler(capputils::ObservableClass*, int) {

}

void AamBuilder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamBuilder();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int spCount = getShapeParameterCount();
  const int tpCount = getTextureParameterCount();
  const int apCount = getAppearanceParameterCount();

  std::vector<boost::shared_ptr<GridModel> >* grids = getGrids().get();
  if (!grids || grids->empty())
    return;

  const int rowCount = grids->at(0)->getRowCount();
  const int columnCount = grids->at(0)->getColumnCount();
  const int pointCount = rowCount * columnCount;
  const int gridCount = grids->size();

  for (unsigned i = 1; i < grids->size(); ++i) {
    if (grids->at(i)->getRowCount() != rowCount ||
        grids->at(i)->getColumnCount() != columnCount)
    {
      return;
    }
  }

  if (!getImages() || getImages()->size() != getGrids()->size())
    return;

  if (getImages()->size() == 0)
    return;

  const unsigned width = getImages()->at(0)->getSize()[0];
  const unsigned height = getImages()->at(0)->getSize()[1];
  const int pixelCount = width * height;
  const int imageCount = getImages()->size();

  if (gridCount != imageCount)
  {
    return;
  }

  boost::shared_ptr<ActiveAppearanceModel> model(new ActiveAppearanceModel());
  model->setShapeParameterCount(getShapeParameterCount());
  model->setTextureParameterCount(getTextureParameterCount());
  model->setAppearanceParameterCount(getAppearanceParameterCount());
  data->setActiveAppearanceModel(model);

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

  model->setMeanShape(meanGrid);

  // Flatten grid points into one large vector
  // Put all vectors together into one large matrix
  // Subtract the mean grid from each grid

  vector<float>* meanGridFeatures = model->getMeanShape().get();
  vector<float> gridFeatureMatrix(2 * pointCount * gridCount);
  for (int i = 0, k = 0; i < gridCount; ++i) {
    boost::shared_ptr<std::vector<float> > features = ActiveAppearanceModel::toFeatures(grids->at(i).get());
    for (int j = 0; j < 2 * pointCount; ++j, ++k) {
      gridFeatureMatrix[k] = features->at(j) - meanGridFeatures->at(j);
    }
  }

  // Calculate principal components of that matrix

  //const int pccols = min(2 * pointCount, gridCount);
  boost::shared_ptr<vector<float> > shapeMatrix(new vector<float> (2 * pointCount * spCount));
  boost::shared_ptr<vector<float> > singularShapeParameters(new vector<float>(spCount));
  culib::getPcs(&(*shapeMatrix)[0], &(*singularShapeParameters)[0], &gridFeatureMatrix[0], 2 * pointCount, gridCount, spCount);

  // The result are the principal grids.
  model->setShapeMatrix(shapeMatrix);
  model->setSingularShapeParameters(singularShapeParameters);

  // Warp all images into reference frame and calculate mean image afterwards
  std::vector<boost::shared_ptr<image_t> > warpedImages;

  ImageWarp warp;
  for (int i = 0; i < imageCount; ++i) {
    if (width != getImages()->at(i)->getSize()[0] ||
        height != getImages()->at(i)->getSize()[1])
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

  boost::shared_ptr<image_t> meanTexture(new image_t(width, height, 1));

  float* buffer = meanTexture->getData();
  for (unsigned i = 0; i < width * height; ++i) {
    float value = 0;
    for (unsigned j = 0; j < warpedImages.size(); ++j) {
      value += warpedImages[j]->getData()[i];
    }
    buffer[i] = value / warpedImages.size();
  }
  model->setMeanTexture(meanTexture);

  // Flatten pixels into one large vector
  // Put all vectors together into one large matrix
  // Subtract the mean image from each image

  vector<float>* meanImageFeatures = model->getMeanTexture().get();
  vector<float> imageFeatureMatrix(pixelCount * imageCount);
  for (int i = 0, k = 0; i < imageCount; ++i) {
    boost::shared_ptr<std::vector<float> > features = ActiveAppearanceModel::toFeatures(warpedImages.at(i).get());
    for (int j = 0; j < pixelCount; ++j, ++k) {
      imageFeatureMatrix[k] = features->at(j) - meanImageFeatures->at(j);
    }
  }

  // Calculate principal components of that matrix

  //const int pcrows = min(pixelCount, imageCount);
  boost::shared_ptr<vector<float> > textureMatrix(new vector<float> (tpCount * pixelCount));
  boost::shared_ptr<vector<float> > singularTextureParameters(new vector<float>(tpCount));
  culib::getPcs(&(*textureMatrix)[0], &(*singularTextureParameters)[0], &imageFeatureMatrix[0], pixelCount, imageCount, tpCount);

  // The result are the principal images.
  model->setTextureMatrix(textureMatrix);
  model->setSingularTextureParameters(singularTextureParameters);

  // calculate shape and texture parameter matrix
  // for each warpedimage/grid
  // - get shape and texture parameters (multiplication of the transpose of the PC matrix)
  // - collect them into a matrix
  // - calculate PCs of that matrix
  // - done

  vector<float> shapeParameterMatrix(gridCount * spCount);
  vector<float> textureParameterMatrix(imageCount * tpCount);
  vector<float> modelFeatureMatrix((spCount + tpCount) * gridCount);

  culib::lintrans(&shapeParameterMatrix[0], &(*shapeMatrix)[0], &gridFeatureMatrix[0], 2 * pointCount, gridCount, spCount, true);
  culib::lintrans(&textureParameterMatrix[0], &(*textureMatrix)[0], &imageFeatureMatrix[0], pixelCount, imageCount, tpCount, true);
  
  // Build concatenated model feature matrix
  for (int iModel = 0, iShape = 0, iTexture = 0, i = 0; i < gridCount; ++i) {
    for (int j = 0; j < spCount; ++j, ++iModel, ++iShape)
      modelFeatureMatrix[iModel] = shapeParameterMatrix[iShape];
    for (int j = 0; j < tpCount; ++j, ++iModel, ++iTexture)
      modelFeatureMatrix[iModel] = textureParameterMatrix[iTexture];
  }

  //const int pcmodelrows = min(spCount + tpCount, gridCount);
  boost::shared_ptr<vector<float> > appearanceMatrix(new vector<float> ((spCount + tpCount) * apCount));
  boost::shared_ptr<vector<float> > singularAppearanceParameters(new vector<float> (apCount));
  culib::getPcs(&(*appearanceMatrix)[0], &(*singularAppearanceParameters)[0], &modelFeatureMatrix[0], spCount + tpCount, gridCount, apCount);

  model->setAppearanceMatrix(appearanceMatrix);
  model->setSingularAppearanceParameters(singularAppearanceParameters);

  // Calculate the gradient matrix
  // - Build the matrix X of texture differences in reference frame
  // - Build the matrix Y of according gradients
  // - For every image grid pair
  //   * change the shape parameters slightly
  //   * this gives the shape gradient y_i
  //   * calculate texture in reference frame as it would appear using the model
  //     and using the modified shape parameters
  //   * The difference from the texture and the image in reference frame gives
  //     the regression variable x_i
  // - Calculate the gradient matrix R = (X'X)^-1X'Y
  // - The gradient is then g = R'x
}

void AamBuilder::writeResults() {
  if (!data)
    return;

  setActiveAppearanceModel(data->getActiveAppearanceModel());
//  setMeanGrid(getActiveAppearanceModel()->getMeanGrid());
  if (getActiveAppearanceModel())
    setMeanImage(getActiveAppearanceModel()->createMeanTexture());
}

}

}
