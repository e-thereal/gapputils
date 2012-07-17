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
#include <sstream>

#include "AamFitter.h"
#include "AamMatchFunction.h"
#include "AamWriter.h"

#include <boost/progress.hpp>

namespace gapputils {

namespace cv {

AamEcdnll::AamEcdnll(boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > trainingSet,
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
  // TODO: why not using the identity?
  newModel->setAppearanceMatrix(oldModel->getAppearanceMatrix());

  // TODO: How to estimate singular values?
  newModel->setSingularShapeParameters(oldModel->getSingularShapeParameters());
  newModel->setSingularTextureParameters(oldModel->getSingularTextureParameters());
  newModel->setSingularAppearanceParameters(oldModel->getSingularAppearanceParameters());
}

AamEcdnll::~AamEcdnll() {
}

boost::shared_ptr<ActiveAppearanceModel> AamEcdnll::updateModel(const DomainType& parameter) {
  const int shapeFeatureCount = newModel->getColumnCount() * newModel->getRowCount() * 2;
  const int shapeMatrixSize = shapeFeatureCount * newModel->getShapeParameterCount();

  assert(shapeFeatureCount + shapeMatrixSize == (int)parameter.size());

  boost::shared_ptr<std::vector<float> > meanShape(new std::vector<float>(shapeFeatureCount));
  boost::shared_ptr<std::vector<float> > shapeMatrix(new std::vector<float>(shapeMatrixSize));

  std::copy(parameter.begin(), parameter.begin() + shapeFeatureCount, meanShape->begin());
  std::copy(parameter.begin() + shapeFeatureCount, parameter.end(), shapeMatrix->begin());
  newModel->setMeanShape(meanShape);
  newModel->setShapeMatrix(shapeMatrix);

  return newModel;
}

double AamEcdnll::eval(const DomainType& parameter) {

  static int iterationCounter = 0;

  // The parameter vector contains the new shape model (mean shape and shape matrix)
  // Create AAM with new shape model and old texture model
  updateModel(parameter);

  AamFitter fitter;
  fitter.setActiveAppearanceModel(newModel);
  fitter.setInReferenceFrame(true);
  fitter.setMeasure(SimilarityMeasure::SSD);
  fitter.setUseAppearanceMatrix(false);

  double ecdnll = 0.0;
  std::cout << "Calculating ecdnll (" << iterationCounter << ")" << std::endl;
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
  if ((iterationCounter++ % parameter.size()) == 0) {
    std::cout << "Saving model..." << std::flush;
    AamWriter writer;
    std::stringstream filename;
    filename << "AAMs/temp_" << (iterationCounter / parameter.size()) << "(" << ecdnll << ").amm";
    writer.setFilename(filename.str());
    writer.setActiveAppearanceModel(newModel);
    writer.execute(0);
    writer.writeResults();
    std::cout << " done!" << std::endl;
  }

  return ecdnll;
}

}

}
