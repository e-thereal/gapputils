/*
 * AamMatchFunction.cu
 *
 *  Created on: Jul 29, 2011
 *      Author: tombr
 */

#include "AamMatchFunction.h"

#include <cassert>
#include <vector>

#include <culib/ICudaImage.h>
#include <culib/CudaImage.h>
#include <culib/lintrans.h>
#include <culib/similarity.h>
#include <culib/transform.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace std;

namespace gapputils {

namespace cv {

namespace cuda {

double aamMatchFunction(const vector<double>& parameter, int spCount, int tpCount, int apCount, int width, int height,
    int columnCount, int rowCount, float* shapeMatrix, float* textureMatrix, float* appearanceMatrix,
    float* meanShape, float* meanTexture,
    culib::ICudaImage* inputImage, culib::ICudaImage* warpedImage,
    bool inReferenceFrame, culib::SimilarityConfig& config, bool useMi)
{
  // Get possible shape and texture parameters for current shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  // - get shape and texture parameters from model parameters

  assert((int)parameter.size() == spCount);

  const int pixelCount = width * height;
  const int pointCount = columnCount * rowCount;

  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);
  vector<float> modelParameters(apCount);

  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  double sim = 0.0;

  copy(parameter.begin(), parameter.end(), shapeParameters.begin());

  culib::lintrans(&shapeFeatures[0], shapeMatrix, &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanShape[i];

  //warp.setInputImage(image);
  //warp.setBaseGrid(model->createShape(&shapeFeatures));
  //warp.setWarpedGrid(model->createMeanShape());
  //warp.execute(0);
  //warp.writeResults();

  thrust::device_vector<float> d_baseGrid(shapeFeatures.begin(), shapeFeatures.end());
  thrust::device_vector<float> d_meanShape(meanShape, meanShape + (2 * pointCount));

  culib::warpImage(warpedImage->getDevicePointer(), inputImage->getCudaArray(), inputImage->getSize(),
        (float2*)d_baseGrid.data().get(), (float2*)d_meanShape.data().get(),
        dim3(columnCount, rowCount));

  warpedImage->saveDeviceToWorkingCopy();

  float* imageFeatures = warpedImage->getWorkingCopy();
  for (int i = 0; i < pixelCount; ++i)
    imageFeatures[i] = imageFeatures[i] - meanTexture[i];
  culib::lintrans(&textureParameters[0], textureMatrix, imageFeatures, pixelCount, 1, tpCount, true);

  // TODO: use appearance matrix. Not used here for debugging purpose
  copy(shapeParameters.begin(), shapeParameters.end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&modelParameters[0], appearanceMatrix, &modelFeatures[0], spCount + tpCount, 1, apCount, true);

  culib::lintrans(&modelFeatures[0], appearanceMatrix, &modelParameters[0], apCount, 1, spCount + tpCount, false);
  copy(modelFeatures.begin(), modelFeatures.begin() + spCount, shapeParameters.begin());
  copy(modelFeatures.begin() + spCount, modelFeatures.end(), textureParameters.begin());

  // Calculate match quality
  // - calculate texture using texture parameters + add mean texture
  // - warp texture to the shape frame
  // - compare both images (SSD)
  culib::lintrans(&textureFeatures[0], textureMatrix, &textureParameters[0], tpCount, 1, pixelCount, false);
  for (int i = 0; i < pixelCount; ++i)
    textureFeatures[i] = textureFeatures[i] + meanTexture[i];

  culib::lintrans(&shapeFeatures[0], shapeMatrix, &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanShape[i];

  if (inReferenceFrame) {
    //warp.setInputImage(image);
    //warp.setBaseGrid(model->createShape(&shapeFeatures));
    //warp.setWarpedGrid(model->createMeanShape());
    //warp.execute(0);
    //warp.writeResults();

    thrust::copy(shapeFeatures.begin(), shapeFeatures.end(), d_baseGrid.begin());
    culib::warpImage(warpedImage->getDevicePointer(), inputImage->getCudaArray(), inputImage->getSize(),
        (float2*)d_baseGrid.data().get(), (float2*)d_meanShape.data().get(),
        dim3(columnCount, rowCount));

    thrust::device_vector<float> d_textureFeatures(textureFeatures.begin(), textureFeatures.end());

    if (!useMi)
      sim = culib::calculateNegativeSSD(warpedImage->getDevicePointer(), d_textureFeatures.data().get(), inputImage->getSize());
    else
      sim = culib::getSimilarity(config, warpedImage->getDevicePointer(), d_textureFeatures.data().get(), inputImage->getSize());
  } else {
    //warp.setInputImage(model->createTexture(&textureFeatures));
    //warp.setBaseGrid(model->createMeanShape());
    //warp.setWarpedGrid(model->createShape(&shapeFeatures));
    //warp.execute(0);
    //warp.writeResults();
    culib::CudaImage textureImage(inputImage->getSize(), inputImage->getVoxelSize(), &textureFeatures[0]);
    thrust::copy(shapeFeatures.begin(), shapeFeatures.end(), d_baseGrid.begin());
    culib::warpImage(warpedImage->getDevicePointer(), textureImage.getCudaArray(), inputImage->getSize(),
        (float2*)d_meanShape.data().get(), (float2*)d_baseGrid.data().get(),
        dim3(columnCount, rowCount));

    if (!useMi)
      sim = culib::calculateNegativeSSD(inputImage->getDevicePointer(), warpedImage->getDevicePointer(), inputImage->getSize());
    else
      sim = culib::getSimilarity(config, inputImage->getDevicePointer(), warpedImage->getDevicePointer(), inputImage->getSize());
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
  return sim;
}

}

}

}
