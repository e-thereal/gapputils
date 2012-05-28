/*
 * CudaImage_gpu.cu
 *
 *  Created on: Feb 9, 2012
 *      Author: tombr
 */

#include "CudaImage.h"

#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

namespace regutil {

class WeightedCoordinate : thrust::binary_function<float, int, float3>
{
private:
  int width, height, depth;

public:
  WeightedCoordinate(int width, int height, int depth) : width(width), height(height), depth(depth) { }

  __host__ __device__
  float3 operator()(const float& value, int index) const {
    const int x = index % width;
    index /= width;
    const int y = index % height;
    index /= height;
    const int z = index;
    return make_float3(x * value, y * value, z * value);
  }

};

float3 CudaImage::getCenterOfGravity() const {
  float3 center = make_float3(0, 0, 0);

  thrust::device_ptr<float> image(getDevicePointer());
  const int width = getSize().x, height = getSize().y, depth = getSize().z;
  const int count = width * height * depth;

  float totalMass = thrust::reduce(image, image + count, 0.f);

  center = thrust::inner_product(image, image + count, thrust::make_counting_iterator(1), center,
      thrust::plus<float3>(), WeightedCoordinate(width, height, depth)) / totalMass;

  center.x *= float(getVoxelSize().x) * 0.001f;
  center.y *= float(getVoxelSize().y) * 0.001f;
  center.z *= float(getVoxelSize().z) * 0.001f;

  return center;
}

}
