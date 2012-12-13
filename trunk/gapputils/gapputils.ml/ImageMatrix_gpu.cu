#include "ImageMatrix.h"

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/math.hpp>
#include <tbblas/repeat.hpp>

#include <cassert>
#include <iostream>

namespace gapputils {

namespace ml {

ImageMatrixChecker::ImageMatrixChecker() {
  ImageMatrix test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(InputImage, test);
  CHECK_MEMORY_LAYOUT2(MinValue, test);
  CHECK_MEMORY_LAYOUT2(MaxValue, test);
  CHECK_MEMORY_LAYOUT2(ColumnCount, test);
  CHECK_MEMORY_LAYOUT2(ImageMatrix, test);
  CHECK_MEMORY_LAYOUT2(AutoScale, test);
  CHECK_MEMORY_LAYOUT2(CenterImages, test);
  CHECK_MEMORY_LAYOUT2(CroppedWidth, test);
  CHECK_MEMORY_LAYOUT2(CroppedHeight, test);
}

void ImageMatrix::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor<float, 3, true>::dim_t dim_t;

  // anneliebttom copy this

  Logbook dlog = getLogbook();

  image_t& inputImage = *getInputImage();

  dim_t inSize = seq(inputImage.getSize()[0], inputImage.getSize()[1], inputImage.getSize()[2]);
  tensor<float, 3, true> input(inSize);
  thrust::copy(inputImage.begin(), inputImage.end(), input.begin());

  int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)inSize[2]));
  int rowCount = ceil((float)inSize[2] / (float)columnCount);

  // centering
  if (getCenterImages()) {
    tensor<float, 3, true> temp = repeat(input, seq(2,2,1));
    input = temp[seq(inSize[0]/2, inSize[1]/2, 0), inSize];
  }
  
  if (getCroppedWidth() > 0 && getCroppedHeight() > 0) {
    dim_t croppedSize = seq(getCroppedWidth(), getCroppedHeight(), inSize[2]);

    tensor<float, 3, true> temp(croppedSize);
    temp = input[(inSize - croppedSize + 1) / 2, croppedSize];
    input = temp;
    inSize = croppedSize;
  }

  float minV = getMinValue();
  float maxV = getMaxValue();

  if (getAutoScale()) {
    minV = min(input);
    maxV = max(input);

    if (monitor)
      dlog() << "Minimum: " << minV << ". Maximum: " << maxV << ".";
  }
  
  dim_t sliceSize = inSize;
  sliceSize[2] = 1;
  dim_t outSize = sliceSize * seq(columnCount, rowCount, 1);

  tensor<float, 3, true> output = zeros<float>(outSize);

  for (int y = 0, z = 0; y < rowCount; ++y) {
    for (int x = 0; x < columnCount && z < inSize[2]; ++x, ++z) {
      output[seq(x, y, 0) * sliceSize, sliceSize] = (input[seq(0, 0, z), sliceSize] - minV) / (maxV - minV);
    }
  }

  boost::shared_ptr<image_t> outputImage(new image_t(outSize[0], outSize[1], outSize[2], inputImage.getPixelSize()));
  thrust::copy(output.begin(), output.end(), outputImage->begin());
  newState->setImageMatrix(outputImage);
}

}

}
