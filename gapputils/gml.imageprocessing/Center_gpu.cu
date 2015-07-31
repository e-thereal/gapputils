/*
 * Center_gpu.cu
 *
 *  Created on: Jul 24, 2015
 *      Author: tombr
 */


#include "Center.h"

#include <tbblas/tensor.hpp>
#include <tbblas/indices.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/math.hpp>
#include <tbblas/io.hpp>

namespace gml {

namespace imageprocessing {

void Center::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::imgproc;

  image_t& input = *getInput();

  tensor<float, 3, true> img(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
  thrust::copy(input.begin(), input.end(), img.begin());

  // Calculate x center
  switch (getMethod()) {
  case CenterMethod::CenterOfGravity:
    {
      const float totalWeight = sum(img);

      float cx = sum(img * repeat(indices<float>(img.size()[0], 1, 1), seq(1, img.size()[1], img.size()[2]))) / totalWeight;
      float cy = sum(img * repeat(indices<float>(1, img.size()[1], 1), seq(img.size()[0], 1, img.size()[2]))) / totalWeight;
      float cz = sum(img * repeat(indices<float>(1, 1, img.size()[2]), seq(img.size()[0], img.size()[1], 1))) / totalWeight;

      if (getRoundToNearest()) {
        newState->setTransform(boost::make_shared<fmatrix4>(make_fmatrix4_translation(
            ::floor(cx - (float)img.size()[0] / 2.0 + 0.5),
            ::floor(cy - (float)img.size()[1] / 2.0 + 0.5),
            ::floor(cz - (float)img.size()[2] / 2.0 + 0.5))));
      } else {
        newState->setTransform(boost::make_shared<fmatrix4>(make_fmatrix4_translation(
            cx - (float)img.size()[0] / 2.0,
            cy - (float)img.size()[1] / 2.0,
            cz - (float)img.size()[2] / 2.0)));
      }
    }
    return;

  case CenterMethod::BestFit:
    {
      img = img > 0;

      // Calculate minimum and maximum x, y, and z.
      float min_x = min(img * repeat(indices<float>(img.size()[0], 1, 1), seq(1, img.size()[1], img.size()[2])) + (img == 0) * img.size()[0]);
      float max_x = max(img * repeat(indices<float>(img.size()[0], 1, 1), seq(1, img.size()[1], img.size()[2])));

      float min_y = min(img * repeat(indices<float>(1, img.size()[1], 1), seq(img.size()[0], 1, img.size()[2])) + (img == 0) * img.size()[1]);
      float max_y = max(img * repeat(indices<float>(1, img.size()[1], 1), seq(img.size()[0], 1, img.size()[2])));

      float min_z = min(img * repeat(indices<float>(1, 1, img.size()[2]), seq(img.size()[0], img.size()[1], 1)) + (img == 0) * img.size()[2]);
      float max_z = max(img * repeat(indices<float>(1, 1, img.size()[2]), seq(img.size()[0], img.size()[1], 1)));

      if (getRoundToNearest()) {
        newState->setTransform(boost::make_shared<fmatrix4>(make_fmatrix4_translation(
            ::floor((max_x + min_x) / 2.0 - (float)img.size()[0] / 2.0 + 0.5),
            ::floor((max_y + min_y) / 2.0 - (float)img.size()[1] / 2.0 + 0.5),
            ::floor((max_z + min_z) / 2.0 - (float)img.size()[2] / 2.0 + 0.5))));
      } else {
        newState->setTransform(boost::make_shared<fmatrix4>(make_fmatrix4_translation(
            (max_x + min_x) / 2.0 - (float)img.size()[0] / 2.0,
            (max_y + min_y) / 2.0 - (float)img.size()[1] / 2.0,
            (max_z + min_z) / 2.0 - (float)img.size()[2] / 2.0)));
      }
    }
    return;
  }
}

} /* namespace imageprocessing */

} /* namespace gml */
