/*
 * util.cpp
 *
 *  Created on: Jul 16, 2012
 *      Author: tombr
 */
#include "util.h"

#include <culib/CudaImage.h>
#include <algorithm>

namespace gapputils {

namespace cv {

boost::shared_ptr<culib::ICudaImage> make_cuda_image(const image_t& image) {
  return boost::shared_ptr<culib::ICudaImage>(new culib::CudaImage(
      dim3(image.getSize()[0], image.getSize()[1], image.getSize()[2]),
      dim3(image.getPixelSize()[0], image.getPixelSize()[1], image.getPixelSize()[2]),
      image.getData()));
}

boost::shared_ptr<image_t> make_gapputils_image(const culib::ICudaImage& image) {

  boost::shared_ptr<image_t> newImage(new image_t(
      image.getSize().x, image.getSize().y, image.getSize().z,
      image.getVoxelSize().x, image.getVoxelSize().y, image.getVoxelSize().z));

  unsigned count = image.getSize().x * image.getSize().y * image.getSize().z;
  std::copy(image.getWorkingCopy(), image.getWorkingCopy() + count, newImage->getData());

  return newImage;
}

}

}
