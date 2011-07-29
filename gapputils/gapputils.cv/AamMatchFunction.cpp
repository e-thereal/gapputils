#include "AamMatchFunction.h"

#include "ImageWarp.h"

#include <algorithm>
#include <cassert>
#include <sstream>

#include <culib/lintrans.h>
#include <culib/similarity.h>
#include <culib/CudaImage.h>
#include <culib/filter.h>

#include <cmath>

#include "FromRgb.h"
#include <gapputils.cv.cuda/AamMatchFunction.h>

using namespace std;
using namespace culib;

namespace gapputils {

namespace cv {

AamMatchFunction::AamMatchFunction(boost::shared_ptr<ICudaImage> image,
    boost::shared_ptr<ActiveAppearanceModel> model, bool inReferenceFrame,
    SimilarityMeasure measure)
 : image(image), model(model), inReferenceFrame(inReferenceFrame), measure(measure)
{
  setupSimilarityConfig(config, dim3(64, 64), make_float2(1.f/64.f, 1.f/64.f));
}

AamMatchFunction::~AamMatchFunction(void)
{
  cleanupSimilarityConfig(config);
}

double AamMatchFunction::eval(const DomainType& parameter) {
  // Get possible shape and texture parameters for current shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  // - get shape and texture parameters from model parameters
  const int apCount = model->getAppearanceParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  assert((int)parameter.size() == spCount);

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);
  vector<float> modelParameters(apCount);
  
  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  double sim = 0.0;

  ImageWarp warp;
  copy(parameter.begin(), parameter.end(), shapeParameters.begin());
  culib::lintrans(&shapeFeatures[0], &(*model->getShapeMatrix())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanShape();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);

  warp.setInputImage(image);
  warp.setBaseGrid(model->createShape(&shapeFeatures));
  warp.setWarpedGrid(model->createMeanShape());
  warp.execute(0);
  warp.writeResults();

  boost::shared_ptr<vector<float> > imageFeatures = model->toFeatures(warp.getOutputImage().get());
  boost::shared_ptr<vector<float> > meanImageFeatures = model->getMeanTexture();
  for (int i = 0; i < pixelCount; ++i)
    (*imageFeatures)[i] = imageFeatures->at(i) - meanImageFeatures->at(i);
  culib::lintrans(&textureParameters[0], &(*model->getTextureMatrix())[0], &(*imageFeatures)[0], pixelCount, 1, tpCount, true);

  // TODO: use appearance matrix. Not used here for debugging purpose
  copy(shapeParameters.begin(), shapeParameters.end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&modelParameters[0], &(*model->getAppearanceMatrix())[0], &modelFeatures[0], spCount + tpCount, 1, apCount, true);

  culib::lintrans(&modelFeatures[0], &(*model->getAppearanceMatrix())[0], &modelParameters[0], apCount, 1, spCount + tpCount, false);
  copy(modelFeatures.begin(), modelFeatures.begin() + spCount, shapeParameters.begin());
  copy(modelFeatures.begin() + spCount, modelFeatures.end(), textureParameters.begin());

  // Calculate match quality
  // - calculate texture using texture parameters + add mean texture
  // - warp texture to the shape frame
  // - compare both images (SSD)
  culib::lintrans(&textureFeatures[0], &(*model->getTextureMatrix())[0], &textureParameters[0], tpCount, 1, pixelCount, false);
  for (int i = 0; i < pixelCount; ++i)
    textureFeatures[i] = textureFeatures[i] + meanImageFeatures->at(i);

  culib::lintrans(&shapeFeatures[0], &(*model->getShapeMatrix())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);

  if (inReferenceFrame) {
    warp.setInputImage(image);
    warp.setBaseGrid(model->createShape(&shapeFeatures));
    warp.setWarpedGrid(model->createMeanShape());
    //warp.setBackgroundImage(background);
    warp.execute(0);
    warp.writeResults();
    boost::shared_ptr<culib::ICudaImage> inputTexture = warp.getOutputImage();
    boost::shared_ptr<culib::ICudaImage> matchTexture = model->createTexture(&textureFeatures);

    if (measure == SSD)
      sim = culib::calculateNegativeSSD(inputTexture->getDevicePointer(), matchTexture->getDevicePointer(), inputTexture->getSize());
    else
      sim = culib::getSimilarity(config, inputTexture->getDevicePointer(), matchTexture->getDevicePointer(), inputTexture->getSize());
  } else {
    boost::shared_ptr<ICudaImage> background(new CudaImage(*image));
    CudaImage kernel(background->getSize());
    createGaussFilter(kernel.getDevicePointer(), kernel.getSize(), 2.0f, kernel.getVoxelSize());
    applyFilter(background->getDevicePointer(), background->getDevicePointer(), kernel.getDevicePointer(), background->getSize());

    warp.setInputImage(model->createTexture(&textureFeatures));
    warp.setBaseGrid(model->createMeanShape());
    warp.setWarpedGrid(model->createShape(&shapeFeatures));
    //warp.setBackgroundImage(background);
    warp.execute(0);
    warp.writeResults();
    boost::shared_ptr<culib::ICudaImage> matchImage = warp.getOutputImage();

    if (measure == SSD)
      sim = culib::calculateNegativeSSD(image->getDevicePointer(), matchImage->getDevicePointer(), image->getSize());
    else
      sim = culib::getSimilarity(config, image->getDevicePointer(), matchImage->getDevicePointer(), image->getSize());
    image->freeCaches();
//    static int iEval = 0;
//    static int iFilename = 0;
//
//    if (iEval % 100 == 0) {
//      std::stringstream filename;
//      filename << "match_" << iFilename++ << " (" << sim << ").jpg";
//
//      FromRgb fromRgb;
//      fromRgb.setRed(matchImage);
//      fromRgb.setGreen(matchImage);
//      fromRgb.setBlue(matchImage);
//      fromRgb.execute(0);
//      fromRgb.writeResults();
//      fromRgb.getImagePtr()->save(filename.str().c_str(), "jpg");
//    }
    //  assert(0);
  }

//  boost::shared_ptr<std::vector<float> > ssp = model->getSingularShapeParameters();
//
//  // penalize large shape variants
//  double pen = 1.0;
//  for (int i = 0; i < spCount; ++i)
//    pen += fabs(parameter[i]) / ssp->at(i);
//
//  return sim - pen;

  boost::shared_ptr<vector<float> > shapeMatrix = model->getShapeMatrix();
  boost::shared_ptr<vector<float> > textureMatrix = model->getTextureMatrix();
  boost::shared_ptr<vector<float> > appearanceMatrix = model->getAppearanceMatrix();

  CudaImage warpedImage(image->getSize(), image->getVoxelSize());

  double sim2 = cuda::aamMatchFunction(parameter, spCount, tpCount, apCount,
      model->getWidth(), model->getHeight(), model->getColumnCount(), model->getRowCount(),
      &(*shapeMatrix)[0], &(*textureMatrix)[0], &(*appearanceMatrix)[0],
      &(*model->getMeanShape())[0], &(*model->getMeanTexture())[0],
      image.get(), &warpedImage, inReferenceFrame, config, (measure == MI));
  assert(sim == sim2);
  return sim;
}

}

}
