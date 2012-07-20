/*
 * util.h
 *
 *  Created on: Jul 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CUDAUTIL_H_
#define GAPPUTILS_ML_CUDAUTIL_H_

#include <culib/ICudaImage.h>

#include <gapputils/Image.h>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace ml {

boost::shared_ptr<culib::ICudaImage> make_cuda_image(const image_t& image);
boost::shared_ptr<image_t> make_gapputils_image(const culib::ICudaImage& image);

}

}

#endif /* UTIL_H_ */
