/*
 * AamEcdnll.cpp
 *
 *  Created on: Jul 20, 2011
 *      Author: tombr
 */

#include "AamEcdnll.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "AamFitter.h"
#include "AamMatchFunction.h"

#include <boost/progress.hpp>

namespace gapputils {

namespace cv {

AamEcdnll::AamEcdnll(boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > > trainingSet,
    boost::shared_ptr<ActiveAppearanceModel> oldModel, float variance, float lambda, std::vector<float>& sigma)
 : trainingSet(trainingSet), oldModel(oldModel), newModel(new ActiveAppearanceModel()), variance(variance), lambda(lambda), sigma(sigma)
{
  assert(trainingSet);
  assert(oldModel);

  newModel->setColumnCount(oldModel->getColumnCount());
  newModel->setRowCount(oldModel->getRowCount());
  newModel->setWidth(oldModel->getWidth());
  newModel->setHeight(oldModel->getHeight());
  newModel->setShapeParameterCount(oldModel->getShapeParameterCount());
  newModel->setTextureParameterCount(oldModel->getTextureParameterCount());

  // This model uses shape and texture parameters directly, thus the appearance matrix
  // is the identity
  //newModel->setAppearanceParameterCount(oldModel->getShapeParameterCount() + oldModel->getTextureParameterCount());
  newModel->setAppearanceParameterCount(oldModel->getAppearanceParameterCount());
  newModel->setMeanTexture(oldModel->getMeanTexture());
  newModel->setTextureMatrix(oldModel->getTextureMatrix());

  // Appearance matrix is mpCount x mpCount identity
  boost::shared_ptr<std::vector<float> > appearanceMatrix(new std::vector<float>(newModel->getAppearanceParameterCount() * newModel->getAppearanceParameterCount()));
  for (int i = 0, k = 0; i < newModel->getAppearanceParameterCount(); ++i) {
    for (int j = 0; j < newModel->getAppearanceParameterCount(); ++j, ++k)
      appearanceMatrix->at(k) = (float)(i == j);
  }
  //newModel->setAppearanceMatrix(appearanceMatrix);
  newModel->setAppearanceMatrix(oldModel->getAppearanceMatrix());
}

AamEcdnll::~AamEcdnll() {
}

double AamEcdnll::eval(const DomainType& parameter) {

  // The parameter vector contains the new shape model (mean shape and shape matrix)
  // Create AAM with new shape model and old texture model
  const int shapeFeatureCount = newModel->getColumnCount() * newModel->getRowCount() * 2;
  const int shapeMatrixSize = shapeFeatureCount * newModel->getShapeParameterCount();

  assert(shapeFeatureCount + shapeMatrixSize == (int)parameter.size());

  boost::shared_ptr<std::vector<float> > meanShape(new std::vector<float>(shapeFeatureCount));
  boost::shared_ptr<std::vector<float> > shapeMatrix(new std::vector<float>(shapeMatrixSize));

  std::copy(parameter.begin(), parameter.begin() + shapeFeatureCount, meanShape->begin());
  std::copy(parameter.begin() + shapeFeatureCount, parameter.end(), shapeMatrix->begin());
  newModel->setMeanShape(meanShape);
  newModel->setShapeMatrix(shapeMatrix);

  AamFitter fitter;
  fitter.setActiveAppearanceModel(newModel);

  double ecdnll = 0.0;
  std::cout << "Calculating ecdnll" << std::endl;
  boost::progress_display showProgress(trainingSet->size());
  for (unsigned i = 0; i < trainingSet->size(); ++i) {
    // Fit current image of the training set with the model
    fitter.setInputImage(trainingSet->at(i));
    fitter.execute(0);
    fitter.writeResults();
    ecdnll -= fitter.getSimilarity();
    // TODO: Calculate parameter penalization term.
    ++showProgress;
  }

  return ecdnll;
}

}

}
