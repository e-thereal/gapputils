#include <capputils/PropertyMap.h>
#include <capputils/Logbook.h>
#include <gapputils/IProgressMonitor.h>
#include <gapputils/Image.h>

#include <tbblas/device_tensor.hpp>

#include <thrust/copy.h>

#include <algorithm>

using namespace tbblas;

namespace gapputils {

namespace cv {

void update_convolve(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor)
{
  image_t& input = *properties.getValue<boost::shared_ptr<image_t> >("InputImage");
  image_t& filter = *properties.getValue<boost::shared_ptr<image_t> >("Filter");
  
  unsigned width = input.getSize()[0], height = input.getSize()[1];
  unsigned fw = filter.getSize()[0], fh = filter.getSize()[1];
  boost::shared_ptr<image_t> output(new image_t(width - fw + 1, height - fh + 1, 1, input.getPixelSize()));
  
  tbblas::device_tensor<float, 3> I(width, height, 1), F(fw, fh, 1), filtered(width - fw + 1, height - fh + 1, 1);
  thrust::copy(input.getData(), input.getData() + (width * height), I.begin());
  thrust::copy(filter.getData(), filter.getData() + (fw * fh), F.begin());
  filtered = conv(I, F);
  
  thrust::copy(filtered.begin(), filtered.end(), output->getData());
  properties.setValue("OutputImage", output);
}

}

}
