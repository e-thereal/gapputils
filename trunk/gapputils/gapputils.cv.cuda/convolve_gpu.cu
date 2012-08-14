#include <capputils/PropertyMap.h>
#include <capputils/Logbook.h>
#include <gapputils/IProgressMonitor.h>
#include <gapputils/Image.h>
#include <capputils/Enumerators.h>

#include <tbblas/device_tensor.hpp>

#include <thrust/copy.h>

#include <algorithm>

using namespace tbblas;

namespace gapputils {

namespace cv {

CapputilsEnumerator(ConvolutionType, Valid, Full, Circular);

void update_convolve(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor)
{
  typedef tbblas::device_tensor<float, 3> tensor_t;
  typedef tbblas::tensor_proxy<tensor_t::iterator, 3> proxy_t;
  
  image_t& input = *properties.getValue<boost::shared_ptr<image_t> >("InputImage");
  image_t& filter = *properties.getValue<boost::shared_ptr<image_t> >("Filter");
  ConvolutionType type = properties.getValue<ConvolutionType>("Type");
  
  const unsigned width = input.getSize()[0], height = input.getSize()[1], depth = input.getSize()[2];
  const unsigned fw = filter.getSize()[0], fh = filter.getSize()[1], fd = filter.getSize()[2];
  
  if (type == ConvolutionType::Valid) {
    boost::shared_ptr<image_t> output(new image_t(width - fw + 1, height - fh + 1, depth - fd + 1, input.getPixelSize()));
    tensor_t I(width, height, depth);
    tensor_t F(fw, fh, fd);
    tensor_t filtered(width - fw + 1, height - fh + 1, depth - fd + 1);
    thrust::copy(input.getData(), input.getData() + (width * height * depth), I.begin());
    thrust::copy(filter.getData(), filter.getData() + (fw * fh * fd), F.begin());
    filtered = conv(I, F);
    
    thrust::copy(filtered.begin(), filtered.end(), output->getData());
    properties.setValue("OutputImage", output);
  } else if (type == ConvolutionType::Full) {
    boost::shared_ptr<image_t> output(new image_t(width + fw - 1, height + fh - 1, depth + fd - 1,
        input.getPixelSize()));
    
    tensor_t padded(width + 2 * fw - 2, height + 2 * fh - 2, depth + 2 * fd - 2);
    tensor_t F(fw, fh, fd);
    tensor_t filtered(width + fw - 1, height + fh - 1, depth + fd - 1);
    tensor_t::dim_t start;
    start[0] = fw - 1;
    start[1] = fh - 1;
    start[2] = fd - 1;
    
    proxy_t paddedProxy = tbblas::subrange(padded, start, input.getSize());
    thrust::copy(input.getData(), input.getData() + (width * height * depth), paddedProxy.begin());
    thrust::copy(filter.getData(), filter.getData() + (fw * fh * fd), F.begin());
  
    filtered = tbblas::conv(padded, F);
    thrust::copy(filtered.begin(), filtered.end(), output->getData());
    properties.setValue("OutputImage", output);
  }
}

}

}
