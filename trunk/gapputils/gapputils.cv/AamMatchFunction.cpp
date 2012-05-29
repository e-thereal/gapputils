#include "AamMatchFunction.h"

#include "ImageWarp.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <iostream>

#include <culib/lintrans.h>
#include <culib/similarity.h>
#include <culib/CudaImage.h>
#include <culib/filter.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <cmath>

#include "FromRgb.h"

using namespace std;
using namespace culib;

namespace gapputils {

namespace cv {

AamMatchFunction::AamMatchFunction(boost::shared_ptr<ICudaImage> image,
    boost::shared_ptr<ActiveAppearanceModel> model, bool inReferenceFrame,
    SimilarityMeasure measure, bool useAm)
 : image(image), model(model), inReferenceFrame(inReferenceFrame), measure(measure), useAm(useAm),
   pointCount(model->getRowCount() * model->getColumnCount()),
   pixelCount(model->getWidth() * model->getHeight()),
   spCount(model->getShapeParameterCount()),
   tpCount(model->getTextureParameterCount()),
   apCount(model->getAppearanceParameterCount()),
   status(spCount, tpCount, apCount, 2 * pointCount, pixelCount, spCount + tpCount),
   warpedImage(new culib::CudaImage(image->getSize(), image->getVoxelSize())),
   d_shapeMatrix(model->getShapeMatrix()->begin(), model->getShapeMatrix()->end()),
   d_textureMatrix(model->getTextureMatrix()->begin(), model->getTextureMatrix()->end()),
   d_appearanceMatrix(model->getAppearanceMatrix()->begin(), model->getAppearanceMatrix()->end()),
   d_meanShape(model->getMeanShape()->begin(), model->getMeanShape()->end()),
   d_meanTexture(model->getMeanTexture()->begin(), model->getMeanTexture()->end())
{
  setupSimilarityConfig(config, dim3(64, 64), make_float2(1.f/64.f, 1.f/64.f));
}

AamMatchFunction::~AamMatchFunction(void)
{
  cleanupSimilarityConfig(config);
}

void saveCudaImage(const std::string& filename, boost::shared_ptr<culib::ICudaImage> image) {
  FromRgb fromRgb;
  fromRgb.setRed(image);
  fromRgb.setGreen(image);
  fromRgb.setBlue(image);
  fromRgb.execute(0);
  fromRgb.writeResults();
  fromRgb.getImagePtr()->save(filename.c_str(), "jpg");
}

//#define OLD_METHOD

double AamMatchFunction::eval(const DomainType& parameter) {
  // Get possible shape and texture parameters for current shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  // - get shape and texture parameters from model parameters
  const int apCount = model->getAppearanceParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  if ((int)parameter.size() != spCount) {
    cout << "parameter.size() = " << parameter.size() << endl;
    cout << "spCount = " << spCount << endl;
    assert((int)parameter.size() == spCount);
  }

#ifdef OLD_METHOD
  double sim = 0.0;
  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);
  vector<float> modelParameters(apCount);
  
  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

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

  boost::shared_ptr<culib::ICudaImage> matchImage;
  boost::shared_ptr<culib::ICudaImage> inputTexture;
  boost::shared_ptr<culib::ICudaImage> matchTexture;

  if (inReferenceFrame) {
    warp.setInputImage(image);
    warp.setBaseGrid(model->createShape(&shapeFeatures));
    warp.setWarpedGrid(model->createMeanShape());
    //warp.setBackgroundImage(background);
    warp.execute(0);
    warp.writeResults();
    inputTexture = warp.getOutputImage();
    matchTexture = model->createTexture(&textureFeatures);

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
    matchImage = warp.getOutputImage();

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
//
//    }
    //  assert(0);
  }
#endif

//  boost::shared_ptr<std::vector<float> > ssp = model->getSingularShapeParameters();
//
//  // penalize large shape variants
//  double pen = 1.0;
//  for (int i = 0; i < spCount; ++i)
//    pen += fabs(parameter[i]) / ssp->at(i);
//
//  return sim - pen;

  // TODO: Need fast method to clear an image
  warpedImage->resetWorkingCopy();

  double sim2 = cuda::aamMatchFunction(parameter, spCount, tpCount, apCount,
      model->getWidth(), model->getHeight(), model->getColumnCount(), model->getRowCount(),
      d_shapeMatrix.data().get(), d_textureMatrix.data().get(), d_appearanceMatrix.data().get(),
      d_meanShape.data().get(), d_meanTexture.data().get(), status,
      image.get(), warpedImage.get(), inReferenceFrame, config, (measure == SimilarityMeasure::MI), useAm);
#ifdef OLD_METHOD
  if(sim != sim2) {
    cout.setf(ios::boolalpha);
    cout << "In reference frame: " << inReferenceFrame << endl;
    cout << "Measure: " << (measure == MI ? "MI" : "SSD") << endl;
    cout << sim << " != " << sim2 << endl;

    if (inputTexture) {
      cout << "Warped Images: " << culib::calculateNegativeSSD(inputTexture->getDevicePointer(), warpedImage->getDevicePointer(), image->getSize()) << endl;
      saveCudaImage("match1.jpg", inputTexture);
    }

    if (matchTexture) {
      cout << "Textures: " << culib::calculateNegativeSSD(matchTexture->getDevicePointer(), status.d_textureFeatures.data().get(), image->getSize()) << endl;
      saveCudaImage("match2.jpg", matchTexture);
    }

    saveCudaImage("match4.jpg", warpedImage);
    thrust::copy(status.d_textureFeatures.begin(), status.d_textureFeatures.end(), thrust::device_ptr<float>(warpedImage->getDevicePointer()));
    saveCudaImage("match3.jpg", warpedImage);
    assert(0);
  }
#endif
  return sim2;
}

}

}
