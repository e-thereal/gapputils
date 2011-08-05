/*
 * AamMatchFunction.cu
 *
 *  Created on: Jul 29, 2011
 *      Author: tombr
 */

#include "AamMatchFunction.h"

#include <cassert>

#include <iostream>
#include <vector>

#include <culib/ICudaImage.h>
#include <culib/CudaImage.h>
#include <culib/lintrans.h>
#include <culib/similarity.h>
#include <culib/transform.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

using namespace std;

namespace gapputils {

namespace cv {

namespace cuda {

#define cout << __LINE__ << endl;

AamMatchStatus::AamMatchStatus(int spCount, int tpCount, int apCount,
    int sfCount, int tfCount, int afCount)
 : shapeParameters(spCount), d_shapeParameters(spCount), d_textureParameters(tpCount), d_appearanceParameters(apCount),
   d_shapeFeatures(sfCount), d_textureFeatures(tfCount), d_appearanceFeatures(afCount)
{ }

double aamMatchFunction(const vector<double>& parameter, int spCount, int tpCount, int apCount, int width, int height,
    int columnCount, int rowCount, float* d_shapeMatrix, float* d_textureMatrix, float* d_appearanceMatrix,
    float* d_meanShape, float* d_meanTexture,
    AamMatchStatus& status,
    culib::ICudaImage* inputImage, culib::ICudaImage* warpedImage,
    bool inReferenceFrame, culib::SimilarityConfig& config, bool useMi, bool useAm)
{
  // Get possible shape and texture parameters for current shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  // - get shape and texture parameters from model parameters

  assert((int)parameter.size() == spCount);

  const int pixelCount = width * height;
  const int pointCount = columnCount * rowCount;

  vector<float>& shapeParameters = status.shapeParameters;
  copy(parameter.begin(), parameter.end(), shapeParameters.begin());

  thrust::device_vector<float>& d_shapeParameters = status.d_shapeParameters;
  thrust::copy(shapeParameters.begin(), shapeParameters.end(), d_shapeParameters.begin());
  thrust::device_vector<float>& d_textureParameters = status.d_textureParameters;
  thrust::device_vector<float>& d_modelParameters = status.d_appearanceParameters;

  thrust::device_vector<float>& d_modelFeatures = status.d_appearanceFeatures;
  thrust::device_vector<float>& d_shapeFeatures = status.d_shapeFeatures;
  thrust::device_vector<float>& d_textureFeatures = status.d_textureFeatures;

  double sim = 0.0;

  culib::lintransDevice(d_shapeFeatures.data().get(), d_shapeMatrix, d_shapeParameters.data().get(), spCount, 1, 2 * pointCount, false);
  thrust::transform(d_shapeFeatures.begin(), d_shapeFeatures.end(), thrust::device_ptr<float>(d_meanShape), d_shapeFeatures.begin(), thrust::plus<float>());

  culib::warpImage(warpedImage->getDevicePointer(), inputImage->getCudaArray(), inputImage->getSize(),
        (float2*)d_shapeFeatures.data().get(), (float2*)d_meanShape,
        dim3(columnCount, rowCount));

  thrust::device_ptr<float> d_imageFeatures(warpedImage->getDevicePointer());
  thrust::transform(d_imageFeatures, d_imageFeatures + pixelCount, thrust::device_ptr<float>(d_meanTexture), d_imageFeatures, thrust::minus<float>());
  culib::lintransDevice(d_textureParameters.data().get(), d_textureMatrix, d_imageFeatures.get(), pixelCount, 1, tpCount, true);

  if (useAm) {
    thrust::copy(d_shapeParameters.begin(), d_shapeParameters.end(), d_modelFeatures.begin());
    thrust::copy(d_textureParameters.begin(), d_textureParameters.end(), d_modelFeatures.begin() + spCount);
    culib::lintransDevice(d_modelParameters.data().get(), d_appearanceMatrix, d_modelFeatures.data().get(), spCount + tpCount, 1, apCount, true);

    culib::lintransDevice(d_modelFeatures.data().get(), d_appearanceMatrix, d_modelParameters.data().get(), apCount, 1, spCount + tpCount, false);
    thrust::copy(d_modelFeatures.begin(), d_modelFeatures.begin() + spCount, d_shapeParameters.begin());
    thrust::copy(d_modelFeatures.begin() + spCount, d_modelFeatures.end(), d_textureParameters.begin());
  }

  // Calculate match quality
  // - calculate texture using texture parameters + add mean texture
  // - warp texture to the shape frame
  // - compare both images (SSD or MI)
  culib::lintransDevice(d_textureFeatures.data().get(), d_textureMatrix, d_textureParameters.data().get(), tpCount, 1, pixelCount, false);
  thrust::transform(d_textureFeatures.begin(), d_textureFeatures.end(), thrust::device_ptr<float>(d_meanTexture), d_textureFeatures.begin(), thrust::plus<float>());

  culib::lintransDevice(d_shapeFeatures.data().get(), d_shapeMatrix, d_shapeParameters.data().get(), spCount, 1, 2 * pointCount, false);
  thrust::transform(d_shapeFeatures.begin(), d_shapeFeatures.end(), thrust::device_ptr<float>(d_meanShape), d_shapeFeatures.begin(), thrust::plus<float>());

  if (inReferenceFrame) {
    // TODO: Need fast method to clear an image
    warpedImage->resetWorkingCopy();

    culib::warpImage(warpedImage->getDevicePointer(), inputImage->getCudaArray(), inputImage->getSize(),
        (float2*)d_shapeFeatures.data().get(), (float2*)d_meanShape,
        dim3(columnCount, rowCount));

    if (!useMi)
      sim = culib::calculateNegativeSSD(warpedImage->getDevicePointer(), d_textureFeatures.data().get(), inputImage->getSize());
    else
      sim = culib::getSimilarity(config, warpedImage->getDevicePointer(), d_textureFeatures.data().get(), inputImage->getSize());
  } else {

    // TODO: Need fast method to clear an image
    warpedImage->resetWorkingCopy();

    // TODO: Need a method to create a CUDA Array from a device pointer
    vector<float> textureFeatures(pixelCount);
    thrust::copy(d_textureFeatures.begin(), d_textureFeatures.end(), textureFeatures.begin());
    culib::CudaImage textureImage(inputImage->getSize(), inputImage->getVoxelSize(), &textureFeatures[0]);

    culib::warpImage(warpedImage->getDevicePointer(), textureImage.getCudaArray(), inputImage->getSize(),
        (float2*)d_meanShape, (float2*)d_shapeFeatures.data().get(),
        dim3(columnCount, rowCount));

    if (!useMi)
      sim = culib::calculateNegativeSSD(inputImage->getDevicePointer(), warpedImage->getDevicePointer(), inputImage->getSize());
    else
      sim = culib::getSimilarity(config, inputImage->getDevicePointer(), warpedImage->getDevicePointer(), inputImage->getSize());
  }

  return sim;
}

}

}

}
