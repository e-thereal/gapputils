#include "AamMatchFunction.h"

#include "ImageWarp.h"

#include <algorithm>
#include <cassert>

#include <culib/lintrans.h>
#include <culib/similarity.h>

using namespace std;

namespace gapputils {

namespace cv {

AamMatchFunction::AamMatchFunction(boost::shared_ptr<culib::ICudaImage> image,
    boost::shared_ptr<ActiveAppearanceModel> model)
 : image(image), model(model)
{
}


AamMatchFunction::~AamMatchFunction(void)
{
}

double AamMatchFunction::eval(const DomainType& parameter) {
  // Get possible shape and texture parameters for current shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  // - get shape and texture parameters from model parameters
  const int mpCount = model->getModelParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  assert(parameter.size() == spCount);

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);
  vector<float> modelParameters(mpCount);
  
  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  ImageWarp warp;
  warp.setWarpedGrid(model->createMeanGrid());
  warp.setInputImage(image);

  copy(parameter.begin(), parameter.end(), shapeParameters.begin());
  culib::lintrans(&shapeFeatures[0], &(*model->getPrincipalGrids())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanGrid();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  warp.setBaseGrid(model->createGrid(&shapeFeatures));
  warp.execute(0);
  warp.writeResults();

  boost::shared_ptr<vector<float> > imageFeatures = model->toFeatures(warp.getOutputImage());
  boost::shared_ptr<vector<float> > meanImageFeatures = model->getMeanImage();
  for (int i = 0; i < pixelCount; ++i)
    (*imageFeatures)[i] = imageFeatures->at(i) - meanImageFeatures->at(i);
  culib::lintrans(&textureParameters[0], &(*model->getPrincipalImages())[0], &(*imageFeatures)[0], pixelCount, 1, tpCount, true);

  copy(shapeParameters.begin(), shapeParameters.end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&modelParameters[0], &(*model->getPrincipalParameters())[0], &modelFeatures[0], spCount + tpCount, 1, mpCount, true);

  culib::lintrans(&modelFeatures[0], &(*model->getPrincipalParameters())[0], &modelParameters[0], mpCount, 1, spCount + tpCount, false);
  copy(modelFeatures.begin(), modelFeatures.begin() + spCount, shapeParameters.begin());
  copy(modelFeatures.begin() + spCount, modelFeatures.end(), textureParameters.begin());

  // Calculate match quality
  // - warp image to reference frame using real shape parameters
  // - calculate texture using texture parameters + add mean texture
  // - compare both textures (SSD)
  culib::lintrans(&shapeFeatures[0], &(*model->getPrincipalGrids())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  warp.setBaseGrid(model->createGrid(&shapeFeatures));
  warp.execute(0);
  warp.writeResults();
  boost::shared_ptr<culib::ICudaImage> inputTexture = warp.getOutputImage();

  culib::lintrans(&textureFeatures[0], &(*model->getPrincipalImages())[0], &textureParameters[0], tpCount, 1, pixelCount, false);
  for (int i = 0; i < pixelCount; ++i)
    textureFeatures[i] = textureFeatures[i] + meanImageFeatures->at(i);
  boost::shared_ptr<culib::ICudaImage> modelTexture = model->createImage(&textureFeatures);

  return culib::calculateNegativeSSD(inputTexture->getDevicePointer(), modelTexture->getDevicePointer(), inputTexture->getSize());
}

}

}
