/*
 * CudaImage.cpp
 *
 *  Created on: Nov 13, 2008
 *      Author: tombr
 */

#include "CudaImage.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

#include <culib/CulibException.h>
#include <culib/transform.h>
#include <culib/bulkmath.h>
#include <culib/filter.h>
#include <culib/util.h>
//#include <DebugUtil/DebugUtil.h>

using namespace culib;
using namespace std;
using namespace optlib;

namespace regutil {

CudaImage::CudaImage(dim3 size, dim3 voxelSize, float* h_originalImage) : h_originalImage(h_originalImage),
    imgDim(size), voxelDim(voxelSize), d_image(0), d_imagePtr(0), d_imageArray(0)
{
  if (!h_originalImage) {
    this->h_originalImage = new float[imgDim.x*imgDim.y*imgDim.z];
    wrapperMode = false;
  } else {
    wrapperMode = true;
  }
  h_workingCopy = new float[imgDim.x*imgDim.y*imgDim.z];
  resetWorkingCopy();
}

CudaImage::CudaImage(const ICudaImage& image) : imgDim(image.getSize()), 
    voxelDim(image.getVoxelSize()), d_image(0), d_imagePtr(0), d_imageArray(0)
{
  const size_t count = imgDim.x*imgDim.y*imgDim.z;
  h_originalImage = new float[count];
  h_workingCopy = new float[count];

  memcpy(h_originalImage, image.getOriginalImage(), sizeof(float) * count);
}

CudaImage::~CudaImage() {
  freeCaches();
  if (h_workingCopy) {
    delete[] h_workingCopy;
    h_workingCopy = 0;
  }

  if (h_originalImage && !wrapperMode) {
    delete[] h_originalImage;
    h_originalImage = 0;
  }
}

void CudaImage::freeCaches() {
  using namespace std;
  if (d_image) {
    CULIB_SAFE_CALL(cudaFree(d_image));
    //cout << "Free device memory: " << (imgDim.x * imgDim.y * imgDim.z) << endl;
  }
  if (d_imagePtr) {
    CULIB_SAFE_CALL(cudaFree(d_imagePtr->ptr));
    delete d_imagePtr;
    //cout << "Free pitched pointer memory: " << (imgDim.x * imgDim.y * imgDim.z) << endl;
  }
  if (d_imageArray) {
    CULIB_SAFE_CALL(cudaFreeArray(d_imageArray));
    //cout << "Free cuda array: " << (imgDim.x * imgDim.y * imgDim.z) << endl;
  }

  d_image = 0;
  d_imagePtr = 0;
  d_imageArray = 0;
}

void CudaImage::fillOriginalImage(const float* h_imageData) {
  memcpy(h_originalImage, h_imageData, sizeof(float) * imgDim.x * imgDim.y * imgDim.z);
  resetWorkingCopy();
}

void CudaImage::saveDeviceToWorkingCopy() {
  if (d_image) {
    CULIB_SAFE_CALL(cudaMemcpy(h_workingCopy, d_image,
        imgDim.x*imgDim.y*imgDim.z*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

void CudaImage::saveDeviceToOriginalImage() {
  if (d_image) {
    CULIB_SAFE_CALL(cudaMemcpy(h_originalImage, d_image,
        imgDim.x*imgDim.y*imgDim.z*sizeof(float), cudaMemcpyDeviceToHost));
    resetWorkingCopy();
  }
}

void CudaImage::resetWorkingCopy() {
  memcpy(h_workingCopy, h_originalImage, imgDim.x*imgDim.y*imgDim.z*sizeof(float));
  freeCaches();
}

void CudaImage::blurImage(double sigma) {
  float *d_kernel;
  const size_t count = imgDim.x*imgDim.y*imgDim.z;

  //cout << "Dim: " << imgDim.x << "x" << imgDim.y << "x" << imgDim.z << endl;

  //printMemoryStats("Start blurring");

  // Blur it if we have to do it
  if (sigma > 0.0) {
    //double start = DebugUtil::getTime();
//    if (sigma * 1000 / (double)getVoxelSize().x > KERNEL_RADIUS / 2 ||
//        sigma * 1000 / (double)getVoxelSize().y > KERNEL_RADIUS / 2 ||
//        sigma * 1000 / (double)getVoxelSize().z > KERNEL_RADIUS / 2)
//    {
      //cout << "FFT blurring" << endl;
      CULIB_SAFE_CALL(cudaMalloc((void**)&d_kernel, count*sizeof(float)));
      createGaussFilter(d_kernel, getSize(), sigma, getVoxelSize());
      applyFilter(getDevicePointer(), getDevicePointer(), d_kernel, getSize());

      // Clean up
      CULIB_SAFE_CALL(cudaFree(d_kernel));
      //cout << "Time vor FFT based method: " << DebugUtil::getTime() - start << endl;
//    } else {
//      //cout << "Convolution based blurring" << endl;
//      //start = DebugUtil::getTime();
//      setGaussKernel(sigma * 1000 / (double)getVoxelSize().x);
//      convolutionRowsGPU(getDevicePointer(), getDevicePointer(), imgDim);
//      if (imgDim.y > 1) {
//        setGaussKernel(sigma * 1000 / (double)getVoxelSize().y);
//        convolutionColumnsGPU(getDevicePointer(), getDevicePointer(), imgDim);
//        if (imgDim.z > 1) {
//          setGaussKernel(sigma * 1000 / (double)getVoxelSize().z);
//          convolutionDepthGPU(getDevicePointer(), getDevicePointer(), imgDim);
//        }
//      }
//      //cout << "Time for convolution based method: " << DebugUtil::getTime() - start << endl;
//    }
  }

  // Save result and clean up
  saveDeviceToWorkingCopy();
  freeCaches();
}

void CudaImage::getXDerivative(ICudaImage& output) const {
  float kernel[KERNEL_LENGTH];
  for (int i = 0; i < KERNEL_LENGTH; ++i)
    if (i == KERNEL_RADIUS-1)
      kernel[i] = -1;
    else if (i == KERNEL_RADIUS+1)
      kernel[i] = 1;
    else
      kernel[i] = 0;
  culib::setConvolutionKernel(kernel);
  convolutionRowsGPU(output.getDevicePointer(), getDevicePointer(), getSize());
}

void CudaImage::getYDerivative(ICudaImage& output) const {
  float kernel[KERNEL_LENGTH];
  for (int i = 0; i < KERNEL_LENGTH; ++i)
    if (i == KERNEL_RADIUS-1)
      kernel[i] = -1;
    else if (i == KERNEL_RADIUS+1)
      kernel[i] = 1;
    else
      kernel[i] = 0;
  culib::setConvolutionKernel(kernel);
  convolutionColumnsGPU(output.getDevicePointer(), getDevicePointer(), getSize());
}

void CudaImage::getZDerivative(ICudaImage& output) const {
  float kernel[KERNEL_LENGTH];
  for (int i = 0; i < KERNEL_LENGTH; ++i)
    if (i == KERNEL_RADIUS-1)
      kernel[i] = -1;
    else if (i == KERNEL_RADIUS+1)
      kernel[i] = 1;
    else
      kernel[i] = 0;
  culib::setConvolutionKernel(kernel);
  convolutionDepthGPU(output.getDevicePointer(), getDevicePointer(), getSize());
}

void CudaImage::adjustImage(const float2& from, const float2& to) {
  ::culib::adjustImage(getDevicePointer(), getDevicePointer(), getSize(), from, to);
  saveDeviceToWorkingCopy();
  freeCaches();
}

/*void CudaImage::equalizeHistogram(uint binCount, float binScale, bool ignoreZeros) {
  ::culib::equalizeHistogram(getDevicePointer(), getDevicePointer(), getSize(), binCount, binScale, ignoreZeros);
  saveDeviceToWorkingCopy();
  freeCaches();
}

void CudaImage::extractMiddleSlice(ICudaImage& slice, const fmatrix4& matrix) const {
//  const size_t count = imgDim.x*imgDim.y*imgDim.z;

  ::culib::extractMiddleSlice(slice.getDevicePointer(), getCudaArray(), getSize(), matrix);
}

void CudaImage::extractMiddleSlice(CudaImage& slice, const fmatrix4& matrix, CudaImage& xshift, CudaImage& yshift, CudaImage& zshift) const {
  ::culib::extractMiddleSlice(slice.getDevicePointer(), getCudaArray(), xshift.getDevicePointer(), yshift.getDevicePointer(), zshift.getDevicePointer(), getSize(), matrix);
}*/

double CudaImage::eval(const int3& point) {
  return 0.0;
}

float* CudaImage::getDevicePointer() const {
  if (!d_image) {
    const size_t size = imgDim.x*imgDim.y*imgDim.z*sizeof(float);
    CULIB_SAFE_CALL(cudaMalloc((void**)&d_image, size));
    CULIB_SAFE_CALL(cudaMemcpy(d_image, h_workingCopy, size, cudaMemcpyHostToDevice));
    //std::cout << "Allocate device memory: " << (imgDim.x * imgDim.y * imgDim.z) << std::endl;
  }
  return d_image;
}

cudaPitchedPtr& CudaImage::getPitchedPointer() const {
  if (!d_imagePtr) {
    d_imagePtr = new cudaPitchedPtr();
    CULIB_SAFE_CALL(cudaMalloc3D(d_imagePtr,
        make_cudaExtent(imgDim.x*sizeof(float), imgDim.y, imgDim.z)));
    linearToPitched(*d_imagePtr, getDevicePointer(), imgDim.x, imgDim.y*imgDim.z);
    //std::cout << "Allocate pitched pointer: " << (imgDim.x * imgDim.y * imgDim.z) << std::endl;
  }

  return *d_imagePtr;
}

cudaArray* CudaImage::getCudaArray() const {
  if (!d_imageArray) {
    const cudaExtent volumeSize = make_cudaExtent(imgDim.x, imgDim.y, imgDim.z);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<float>();
    CULIB_SAFE_CALL(cudaMalloc3DArray(&d_imageArray, &texDesc, volumeSize));

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent = volumeSize;
    cpy_params.kind = cudaMemcpyHostToDevice;
    cpy_params.dstArray = d_imageArray;
    cpy_params.srcPtr = make_cudaPitchedPtr(h_workingCopy, imgDim.x*sizeof(float),
        imgDim.x, imgDim.y );
    CULIB_SAFE_CALL(cudaMemcpy3D(&cpy_params));

    CULIB_SAFE_CALL(cudaThreadSynchronize());
    //std::cout << "Allocate cuda array: " << (imgDim.x * imgDim.y * imgDim.z) << endl;
  }
  return d_imageArray;
}

float* CudaImage::getWorkingCopy() const {
  return h_workingCopy;
}

float* CudaImage::getOriginalImage() const {
  return h_originalImage;
}

const dim3& CudaImage::getSize() const {
  return imgDim;
}

const dim3& CudaImage::getVoxelSize() const {
  return voxelDim;
}

void CudaImage::setParameter(int id, double value) {
  switch(id) {
    case BlurringSigma: blurImage(value); break;
    case FreeCaches: freeCaches(); break;
    case ResetWorkingCopy: resetWorkingCopy(); break;
    case MinFrom: from.x = value; break;
    case MaxFrom: from.y = value; break;
    case MinTo:   to.x = value; break;
    case MaxTo:   to.y = value; adjustImage(from, to); break;
  }
}

void CudaImage::setParameter(int id, void* value) {
}

}
