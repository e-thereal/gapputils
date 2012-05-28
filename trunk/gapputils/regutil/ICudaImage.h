/**
 * @file ICudaImage.h
 * @brief Basic methods all CudeImages should provide
 *
 * @date Nov 13, 2008
 * @author Tom Brosch
 */

#ifndef _REGUTIL_ICUDAIMAGE_H_
#define _REGUTIL_ICUDAIMAGE_H_

#include <reglib/RegistrationProblem.h>
#include <optlib/IParameterizable.h>
#include <cuda_runtime.h>
//#include <culib/math3d.h>

namespace regutil {

/// Tells, what functions a CudaImage should provide
/**
 * The idea is, that the image data is always located in host memory. When you
 * need  the image  as  a  device pointer,  a pitched pointer or a cuda array,
 * the image data  is copied from the host to the device.  To speed up things,
 * there are copies created on the device, which can be written back to the host
 * when necessary.
 *
 * @remarks
 *  - I want to discourage from modifying  the image data using any device re-
 *    presentation,  because those changes  are never written back to the host
 *    and likely lost,  when the device representation is overwritten with the
 *    host memory copy.
 *  - When calling any getter, the class is responsible to keep the device re-
 *    presentations consistent  with  the working copy.  Therefore,  automatic
 *    memory allocations and copy operations are performed when necessary.
 *  - To save space you can use the freeCaches() method, to free the memory of
 *    all cached device pointers.
 *  - All device pointers are temporary  and should not be stored,  since they
 *    will become invalid after a call of freeCaches().
 */
class ICudaImage : public virtual reglib::IImageFunction<int3>,
                   public virtual optlib::IParameterizable
{
public:

  /// Fills the original image
  virtual void fillOriginalImage(const float* h_imageData) = 0;

  /// Get the image data as a device pointer
  /**
   * @return A pointer to the image data on the device
   */
  virtual float* getDevicePointer() const = 0;

  /// Get the image data as a pitched pointer
  /**
   * This functions creates a pitched pointer on the device and returns the pointer
   *
   * @return The created pitched pointer
   */
  virtual cudaPitchedPtr& getPitchedPointer() const = 0;

  /// Get the image data as a CUDA array
  /**
   * This function creates a CUDA array on the device and returns a handle to it
   *
   * @return A handle to the created CUDA array
   */
  virtual cudaArray* getCudaArray() const = 0;

  /// Get the pointer to the working copy in host memory
  /**
   * @ return The pointer to the working copy in host memory
   */
  virtual float* getWorkingCopy() const = 0;

  /// Get the pointer to the original image in host memory
  /**
   * @ return The pointer to the original image in host memory
   */
  virtual float* getOriginalImage() const = 0;

  /// Get the width, height and depth of the image
  /**
   * @return The image dimensions in number of voxels
   */
  virtual const dim3& getSize() const = 0;

  /// Get the width, height and depth of a voxel
  /**
   * @return The voxel dimension in micrometers
   */
  virtual const dim3& getVoxelSize() const = 0;

  /// Frees all device caches
  /**
   * This function frees all memory, that was allocated on the device
   */
  virtual void freeCaches() = 0;

  /// Resets the working copy
  /**
   * This method overriddes the working copy with the original image data
   */
  virtual void resetWorkingCopy() = 0;

  /// Extracts a slice from a volume
  /**
   */
  //virtual void extractMiddleSlice(ICudaImage& slice, const fmatrix4& matrix) const = 0;
};

}

#endif /* _REGUTIL_ICUDAIMAGE_H_ */
