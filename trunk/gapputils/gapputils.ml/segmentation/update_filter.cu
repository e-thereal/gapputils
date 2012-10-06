/*
 * Trainer_gpu.cu
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#include <capputils/PropertyMap.h>
#include <capputils/Logbook.h>
#include <gapputils/IProgressMonitor.h>
#include <gapputils/Image.h>

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>
#include <thrust/copy.h>

#include <tbblas/fft.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/min.hpp>
#include <tbblas/max.hpp>
#include <tbblas/math.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/binary_expression.hpp>

namespace gapputils {

namespace ml {

namespace segmentation {

typedef tbblas::tensor<double, 3, false> host_volume;
typedef tbblas::tensor<double, 3, true> volume;
typedef tbblas::tensor<tbblas::complex<double>, 3, true> cvolume;
typedef typename volume::dim_t dim_t;

void update_filter(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor)
{
  using namespace tbblas;
  using capputils::Severity;

  image_t& input = *properties.getValue<boost::shared_ptr<image_t> >("Image");
  std::vector<boost::shared_ptr<host_volume> >& tensors =
      *properties.getValue<boost::shared_ptr<std::vector<boost::shared_ptr<host_volume> > > >("Tensors");

  dim_t imageSize = sequence<unsigned, 3>(input.getSize());

  volume image(imageSize), paddedFilter, filter(tensors[0]->size()), centeredFilter, filtered, stddev;
  thrust::copy(input.getData(), input.getData() + image.count(), image.begin());

  cvolume cimage = fft(image), cfilter, cfiltered;

  boost::shared_ptr<image_t> padded(new image_t(input.getSize()[0], input.getSize()[1], input.getSize()[2] * tensors.size()));
  boost::shared_ptr<image_t> flipped(new image_t(input.getSize()[0], input.getSize()[1], input.getSize()[2] * tensors.size()));
  boost::shared_ptr<image_t> centered(new image_t(input.getSize()[0], input.getSize()[1], input.getSize()[2] * tensors.size()));
  boost::shared_ptr<image_t> output(new image_t(input.getSize()[0], input.getSize()[1], input.getSize()[2] * tensors.size()));

  dim_t topleft = (image.size() - filter.size() + 1u) / 2u;

  // normalize the image
  volume meanFilter = zeros<double>(image.size());
  meanFilter[topleft, filter.size()] = ones<double>(filter.size()) / (double)filter.count();

  volume shiftedFilter = ifftshift(meanFilter);
  cfilter = fft(shiftedFilter);
  cfiltered = cimage * cfilter;
  filtered = ifft(cfiltered);
  stddev = (image - filtered) * (image - filtered);

  cimage = fft(stddev);
  cfiltered = cimage * cfilter;
  filtered = ifft(cfiltered);
  stddev = sqrt(filtered);

  cimage = fft(image);

  for (unsigned i = 0; i < tensors.size(); ++i) {
    thrust::copy(stddev.begin(), stddev.end(), padded->getData() + i * image.count());

    thrust::copy(tensors[i]->begin(), tensors[i]->end(), filter.begin());
    filter = filter - sum(filter) / (double)filter.count();
    filter = filter / sqrt(sum(filter * filter) / (double)filter.count());

    paddedFilter = zeros<double>(image.size());
    paddedFilter[topleft, filter.size()] = filter;
    thrust::copy(paddedFilter.begin(), paddedFilter.end(), centered->getData() + i * image.count());

    paddedFilter[topleft, filter.size()] = flip(filter);
    centeredFilter = ifftshift(paddedFilter);

    cfilter = fft(centeredFilter);
    cfiltered = cimage * cfilter;
    filtered = ifft(cfiltered);

    double minimum = min(filtered), maximum = max(filtered);
    if (i == 0)
      logbook(Severity::Trace) << "min: " << minimum << "; max: " << minimum;

//    filtered = (filtered - minimum) / (maximum - minimum);
    filtered = filtered / stddev;
//    filtered = filtered == ones<double>(filtered.size()) * max(filtered);
    thrust::copy(filtered.begin(), filtered.end(), output->getData() + i * filtered.count());
  }
  properties.setValue<boost::shared_ptr<image_t> >("Padded", padded);
  properties.setValue<boost::shared_ptr<image_t> >("Centered", centered);
  properties.setValue<boost::shared_ptr<image_t> >("Output", output);
}

}

}

}
