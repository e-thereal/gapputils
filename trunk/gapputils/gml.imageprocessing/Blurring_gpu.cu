#include "Blurring.h"

#include <tbblas/tensor.hpp>
#include <tbblas/gaussian.hpp>
#include <tbblas/fft.hpp>

namespace gml {

namespace imageprocessing {

BlurringChecker::BlurringChecker() {
  Blurring blurring;
  blurring.initializeClass();

  CHECK_MEMORY_LAYOUT2(InputImage, blurring);
  CHECK_MEMORY_LAYOUT2(Sigma, blurring);
  CHECK_MEMORY_LAYOUT2(OutputImage, blurring);
}

void Blurring::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  image_t& image = *getInputImage();

  tensor<float, 3, true> input(image.getSize()[0], image.getSize()[1], image.getSize()[2]), filter, output;
  thrust::copy(image.begin(), image.end(), input.begin());

  filter = gaussian<float>(input.size(), getSigma());
  tensor<complex<float>, 3, true> cinput, cfilter, coutput;
  tensor<cufftComplex, 3, true> test;
  /*cinput = fft(input);
  cfilter = fft(filter);
  coutput = cinput * cfilter;
  output = ifft(coutput);

  boost::shared_ptr<image_t> outputImage(new image_t(image.getSize(), image.getPixelSize()));
  thrust::copy(output.begin(), output.end(), outputImage->begin());
  newState->setOutputImage(outputImage);*/
}

}

}
