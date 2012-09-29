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
#include <tbblas/tensor_base.hpp>
#include <tbblas/zeros.hpp>
#include <thrust/copy.h>

#include <tbblas/fft2.hpp>
#include <tbblas/flip.hpp>

namespace gapputils {

namespace ml {

namespace segmentation {

typedef tbblas::tensor<double, 3, true> volume;
typedef tbblas::tensor<tbblas::complex<double>, 3, true> cvolume;
typedef tbblas::tensor_base<double, 3, false> tensor_t;

void update_trainer(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor)
{
  using namespace tbblas;

  logbook() << "Doing something.";
  image_t& input = *properties.getValue<boost::shared_ptr<image_t> >("Image");
  std::vector<boost::shared_ptr<tensor_t> >& tensors =
      *properties.getValue<boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > >("Tensors");

  volume image(input.getSize());
  thrust::copy(input.getData(), input.getData() + image.count(), image.begin());

  cvolume fimage = fft(image);

  boost::shared_ptr<image_t> output(new image_t(input.getSize()[0], input.getSize()[1], input.getSize()[2] * tensors.size()));
  for (unsigned i = 0; i < tensors.size(); ++i) {
    volume filter = zeros<double, 3>(image.size());
    thrust::copy(tensors[i]->begin(), tensors[i]->end(),
        flip(filter[seq(0u,0u,0u), sequence<unsigned,3>(tensors[i]->size())]).begin());

    cvolume ffilter = fft(filter);
    cvolume ffiltered = fimage * ffilter;
    volume filtered = ifft(ffiltered);

    thrust::copy(filtered.begin(), filtered.end(), output->getData() + i * filtered.count());
  }
  properties.setValue<boost::shared_ptr<image_t> >("Output", output);
}

}

}

}
