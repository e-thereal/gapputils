/*
 * AamUtils.cpp
 *
 *  Created on: Jul 15, 2011
 *      Author: tombr
 */

#include "AamUtils.h"

#include <cassert>

#include <culib/lintrans.h>
#include <cassert>
#include "ImageWarp.h"

using namespace std;

namespace gapputils {

namespace cv {

void AamUtils::getAppearanceParameters(std::vector<float>* appearanceParameters,
      ActiveAppearanceModel* model, GridModel* grid, image_t* image)
{
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();
  const int apCount = model->getAppearanceParameterCount();
  const int pixelCount = image->getSize()[0] * image->getSize()[1] * image->getSize()[2];

  boost::shared_ptr<vector<float> > shapeFeatures = model->toFeatures(grid);
  vector<float>* meanGridFeatures = model->getMeanShape().get();
  vector<float> shapeParameters(spCount);

  assert(shapeFeatures->size() == meanGridFeatures->size());
  assert((int)appearanceParameters->size() == apCount);

  for (unsigned i = 0; i < shapeFeatures->size(); ++i)
    (*shapeFeatures)[i] -= meanGridFeatures->at(i);
  culib::lintrans(&shapeParameters[0], &(*model->getShapeMatrix())[0], &(*shapeFeatures)[0], shapeFeatures->size(), 1, spCount, true);

  vector<float> textureParameters(tpCount);
  ImageWarp warp;
  warp.setInputImage(model->createTexture(model->toFeatures(image).get()));
  warp.setBaseGrid(model->createShape(shapeFeatures.get()));
  warp.setWarpedGrid(model->createMeanShape());
  warp.execute(0);
  warp.writeResults();

  boost::shared_ptr<vector<float> > imageFeatures = model->toFeatures(warp.getOutputImage().get());
  boost::shared_ptr<vector<float> > meanImageFeatures = model->getMeanTexture();
  for (int i = 0; i < pixelCount; ++i)
    (*imageFeatures)[i] = imageFeatures->at(i) - meanImageFeatures->at(i);
  culib::lintrans(&textureParameters[0], &(*model->getTextureMatrix())[0], &(*imageFeatures)[0], pixelCount, 1, tpCount, true);

  vector<float> modelFeatures(spCount + tpCount);
  copy(shapeParameters.begin(), shapeParameters.end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&(*appearanceParameters)[0], &(*model->getAppearanceMatrix())[0], &modelFeatures[0], spCount + tpCount, 1, apCount, true);
}

void AamUtils::getShapeParameters(std::vector<float>* shapeParameters,
      ActiveAppearanceModel* model, std::vector<float>* appearanceParameters)
{
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();
  const int apCount = model->getAppearanceParameterCount();
  vector<float> appearanceFeatures(spCount + tpCount);
  culib::lintrans(&appearanceFeatures[0], &(*model->getAppearanceMatrix())[0], &(*appearanceParameters)[0], apCount, 1, spCount + tpCount, false);
  copy(appearanceFeatures.begin(), appearanceFeatures.begin() + spCount, shapeParameters->begin());
}

void AamUtils::getShapeFeatures(std::vector<float>* shapeFeatures,
      ActiveAppearanceModel* model, std::vector<float>* shapeParameters)
{
  assert(shapeFeatures->size() == model->getMeanShape()->size());
  culib::lintrans(&(*shapeFeatures)[0], &(*model->getShapeMatrix())[0], &(*shapeParameters)[0], model->getShapeParameterCount(), 1, shapeFeatures->size(), false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanShape();
  for (unsigned i = 0; i < shapeFeatures->size(); ++i)
    (*shapeFeatures)[i] = (*shapeFeatures)[i] + meanGrid->at(i);
}

}

}
