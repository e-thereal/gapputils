/*
 * util.h
 *
 *  Created on: Jul 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_UTIL_H_
#define GAPPUTILS_CV_UTIL_H_

#include <culib/CudaImage.h>

#include <gapputils/Image.h>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace cv {

// TODO: only defined if CUDA support is active
boost::shared_ptr<ICudaImage> make_cuda_image(const image_t& image) {
  return boost::shared_ptr<ICudaImage>(new culib::CudaImage(
      dim3(image.getSize()[0], image.getSize()[1], image.getSize()[2]),
      dim3(image.getPixelSize()[0], image.getPixelSize()[1], image.getPixelSize()[2]),
      image.getData()));
}

}

}

#endif /* UTIL_H_ */
