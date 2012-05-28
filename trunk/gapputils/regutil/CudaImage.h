/**
 * @file CudaImage.h
 * @brief Contains the CudaImage class
 *
 *  Created on: Nov 13, 2008
 *      Author: tombr
 */

#ifndef _REGUTIL_CUDAIMAGE_H_
#define _REGUTIL_CUDAIMAGE_H_

#include "regutil.h"

#include <culib/math3d.h>
#include "ICudaImage.h"

namespace regutil {

/// A ready to use implementation of the ICudaImage interface
/**
 * This implementation stores to versions of the image on the host, the original
 * image and a working copy.
 *
 * @remarks
 *
 *  - When you load an image, the original image and a working copy resides in
 *    host memory
 *  - Blurring is always done to the working copy
 *  - The freeCaches() method  is automatically called  whenever the
 *    the working copy  is changed,  to force  a renew of all caches  the next
 *    time a device pointer is requested.
 *  - Please read the remarks of ICudaImage
 */
class CudaImage : public virtual ICudaImage {
protected:
  float* h_workingCopy;         ///< Host pointer to the working copy
  float* h_originalImage;       ///< Host pointer to the original image
  dim3 imgDim;                  ///< Image dimensions in number of voxel
  dim3 voxelDim;                ///< Voxel dimensions in millimeters
  int wrapperMode;              ///< True, iff this class is used to wrap another image
  float2 from;                  ///< The from interval for the image adjustment
  float2 to;                    ///< The to interval for the image adjustment

private:
  mutable float* d_image;
  mutable cudaPitchedPtr* d_imagePtr;
  mutable cudaArray *d_imageArray;

public:

  /// Constructor to creates a CudaImage
  /**
   * @param[in] size      (Optional) The size of the image in voxel (Default: (1,1,1))
   * @param[in] voxelSize (Optional) The size of a voxel in millimeters (Default: (1,1,1))
   *
   * @remark
   * If an original image is given the new object wraps around this image. This means:
   * - No memory for the original image in host memory is allocated
   * - No memory of the original is deallocated while deleting the cuda image
   */
  CudaImage(dim3 size = dim3(), dim3 voxelSize = dim3(), float* h_originalImage = 0);

  /// Copy constructor
  CudaImage(const ICudaImage& image);

  /// Virtual Destructor
  virtual ~CudaImage();

  /// Perform Gaussian Blurring with a given sigma
  /**
   * @remarks
   * - Before blurring the image
   * - Blurring is applied to the working copy only
   *
   * @param[in] sigma The sigma value, used to create the Gaussian blurring filter
   */
  void blurImage(double sigma);

  ///
  void getXDerivative(ICudaImage& output) const;
  void getYDerivative(ICudaImage& output) const;
  void getZDerivative(ICudaImage& output) const;

  /// Adjusts the pixel values of the image
  /**
   * @param[in]   from      Minimum and maximum value of the range, that will be used for mapping pixel values
   * @param[out]  to        Minimum and maximum value of the range, where all pixel values will be mapped to
   *
   *  This function maps all values from the interval [from.x, from.y] to the interval [to.x, to.y]
   */
  void adjustImage(const float2& from, const float2& to);

  /// Performs a histogram equalization
  /**
   * @param[out]  d_result  Allocated memory for the result. If d_result == d_image, the operation is done inplace
   * @param[in]   d_image   The input image in device memory
   * @param[in]   dimension The dimension of the image
   * @param[in]   ignoreZeros (Optional) If set to true, bin 0 is set to the value of bin 1 before the histogram equalization. Default false.
   *
   * @remarks
   * - Set ignoreZeros to true if you have large black areas in our image in order to ensure those parts remain black in the
   *   filtered image.
   */
  //void equalizeHistogram(uint binCount, float binScale, bool ignoreZeros = false);

  /// Fast method that does not save any results back to the host
  //virtual void extractMiddleSlice(ICudaImage& slice, const fmatrix4& matrix) const;

  //void extractMiddleSlice(CudaImage& slice, const fmatrix4& matrix, CudaImage& xshift, CudaImage& yshift, CudaImage& zshift) const;

  float3 getCenterOfGravity() const;

  virtual double eval(const int3& point);

  virtual void fillOriginalImage(const float* h_imageData);
  virtual float* getDevicePointer() const;
  virtual cudaPitchedPtr& getPitchedPointer() const;
  virtual cudaArray* getCudaArray() const;
  virtual float* getWorkingCopy() const;
  virtual float* getOriginalImage() const;

  virtual const dim3& getSize() const;
  virtual const dim3& getVoxelSize() const;

  virtual void setParameter(int id, double value);
  virtual void setParameter(int id, void* value);

  virtual void freeCaches();
  virtual void resetWorkingCopy();

  /// Writes memory from the device back to the working copy on the host
  /**
   * @remarks
   * - This is only done, when getDevicePointer() has been called beforehand
   */
  void saveDeviceToWorkingCopy();

  /// Writes memory from the device back to the original image on the host
  /**
   * @remarks
   * - This is only done, when getDevicePointer() has been called beforehand
   */
  void saveDeviceToOriginalImage();
};

}

#endif /* _REGUTIL_CUDAIMAGE_H_ */
