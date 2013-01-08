/*
 * VolumeRenderer_gpu.cu
 *
 *  Created on: Jan 7, 2013
 *      Author: tombr
 */

#include "VolumeRenderer.h"

#include <thrust/functional.h>

#include <tbblas/tensor.hpp>

#include "math3d.h"

namespace gml {

namespace imageprocessing {

VolumeRendererChecker::VolumeRendererChecker() {
  VolumeRenderer test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Volume, test);
  CHECK_MEMORY_LAYOUT2(Distance, test);
  CHECK_MEMORY_LAYOUT2(Angle, test);
  CHECK_MEMORY_LAYOUT2(SampleCount, test);
  CHECK_MEMORY_LAYOUT2(Mode, test);
  CHECK_MEMORY_LAYOUT2(Orientation, test);
  CHECK_MEMORY_LAYOUT2(Image, test);
}

texture<float, 3, cudaReadModeElementType> tex;  // 3D texture

__global__ void volumeRenderKernel(float* result, int width, int height, int depth,
    float distance, float angle, int sampleCount, int renderMode)
{
  uint i = blockDim.x*blockIdx.x+threadIdx.x;
  uint j = blockDim.y*blockIdx.y+threadIdx.y;

  if (i >= width || j >= height)
    return;

  float vx = (float)width / 2.0;
  float vy = (float)height / 2.0;

  float x = (float)i + 0.5;
  float y = (float)j + 0.5;
  float z = (float)depth / 2.0;

  float dx = (vx - x) / distance;
  float dy = (vy - y) / distance;

  fmatrix4 transform =
      make_fmatrix4_translation((float)width / 2.0f, (float)height / 2.0f, (float)depth / 2.0f) *
      make_fmatrix4_rotationY(angle) *
      make_fmatrix4_translation((float)-width / 2.0f, (float)-height / 2.0f, (float)-depth / 2.0f);

  switch (renderMode) {
  case VolumeRenderMode::MaximumIntensityProjection:
    {
      float4 v = transform * make_float4(x, y, z, 1);
      float maximum = tex3D(tex, get_x(v), get_y(v), get_z(v));

      for (int k = 0; k < sampleCount; ++k) {
        float t = (float)k / (float)(sampleCount - 1) - 0.5;
        v = transform * make_float4(x + t * dx, y + t * dy, z + t * depth, 1);
        maximum = max(maximum, tex3D(tex, get_x(v), get_y(v), get_z(v)));
      }
      result[j * width + i] = maximum;
    }
    break;

  case VolumeRenderMode::AverageProjection:
    {
      float sum = 0.0f;
      float4 v;
      for (int k = 0; k < sampleCount; ++k) {
        float t = (float)k / (float)(sampleCount - 1) - 0.5;
        v = transform * make_float4(x + t * dx, y + t * dy, z + t * depth, 1);
        sum += tex3D(tex, get_x(v), get_y(v), get_z(v));
      }
      result[j * width + i] = sum / sampleCount;
    }
    break;
  }
}

void VolumeRenderer::update(IProgressMonitor* monitor) const {
  const int BlockWidth = 32;
  const int BlockHeight = 32;

  image_t& volume = *getVolume();
  const int width = volume.getSize()[0];
  const int height = volume.getSize()[1];
  const int depth = volume.getSize()[2];

  // Load image into texture memory
  cudaArray *d_array;

  const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
  cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&d_array, &texDesc, volumeSize);

  cudaMemcpy3DParms cpy_params = {0};
  cpy_params.extent = volumeSize;
  cpy_params.kind = cudaMemcpyHostToDevice;
  cpy_params.dstArray = d_array;
  cpy_params.srcPtr = make_cudaPitchedPtr(volume.getData(), width * sizeof(float), width, height);
  cudaMemcpy3D(&cpy_params);

  // set texture parameters
  tex.normalized = false;                   // access with normalized texture coordinates
  tex.filterMode = cudaFilterModeLinear;    // linear interpolation
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.addressMode[2] = cudaAddressModeClamp;

  // bind array to 3D texture
  cudaBindTextureToArray(tex, d_array, texDesc);

  tbblas::tensor<float, 2, true> image(width, height);

  // Start kernel
  dim3 gridDim((width + BlockWidth - 1) / BlockWidth, (height + BlockHeight - 1) / BlockHeight);
  dim3 blockDim(BlockWidth, BlockHeight);
  volumeRenderKernel<<<gridDim, blockDim>>>(image.data().data().get(), width, height, depth,
      getDistance(), getAngle() * M_PI / 180.0, getSampleCount(), getMode());

  cudaStreamSynchronize(0);
  cudaUnbindTexture(tex);
  cudaFreeArray(d_array);

  boost::shared_ptr<image_t> output(new image_t(width, height, 1));
  thrust::copy(image.begin(), image.end(), output->begin());
  newState->setImage(output);
}

}

}
