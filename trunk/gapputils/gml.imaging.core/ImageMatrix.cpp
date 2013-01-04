/*
 * ImageMatrix.cpp
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#include "ImageMatrix.h"

#include <capputils/EventHandler.h>
//#include <capputils/Verifier.h>
#include <capputils/TimeStampAttribute.h>

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/math.hpp>
#include <tbblas/repeat.hpp>

#include <cmath>

namespace gml {

namespace imaging {

namespace core {

int ImageMatrix::inputId;

BeginPropertyDefinitions(ImageMatrix)

  ReflectableBase(DefaultWorkflowElement<ImageMatrix>)

  WorkflowProperty(InputImage, Input("In"), NotNull<Type>(), TimeStamp(inputId = Id))
  WorkflowProperty(ImageMatrix, Output("Out"))
  WorkflowProperty(MaxSliceCount)
  WorkflowProperty(MinValue)
  WorkflowProperty(MaxValue)
  WorkflowProperty(AutoScale)
  WorkflowProperty(ColumnCount,
      Description("The number of columns. A value of -1 indicates to always use a squared matrix."))
  WorkflowProperty(CenterImages)
  WorkflowProperty(CroppedWidth)
  WorkflowProperty(CroppedHeight)

EndPropertyDefinitions

ImageMatrix::ImageMatrix() : _MaxSliceCount(-1), _MinValue(-2), _MaxValue(2), _AutoScale(false),
 _ColumnCount(-1), _CenterImages(false), _CroppedWidth(-1), _CroppedHeight(-1)
{
  setLabel("Matrix");
  Changed.connect(capputils::EventHandler<ImageMatrix>(this, &ImageMatrix::changedHandler));
}

void ImageMatrix::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && Verifier::Valid(*this)) {
    execute(0);
    writeResults();
  }
}

void ImageMatrix::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor<float, 3>::dim_t dim_t;

  // anneliebttom copy this

  Logbook dlog = getLogbook();

  image_t& inputImage = *getInputImage();

  dim_t inSize = seq(inputImage.getSize()[0], inputImage.getSize()[1],
      getMaxSliceCount() > 0 ? std::min(getMaxSliceCount(), (int)inputImage.getSize()[2]) : inputImage.getSize()[2]);
  tensor<float, 3> input(inSize);
  thrust::copy(inputImage.begin(), inputImage.begin() + input.count(), input.begin());

  int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)inSize[2]));
  int rowCount = ceil((float)inSize[2] / (float)columnCount);

  // centering
  if (getCenterImages()) {
    tensor<float, 3> temp = repeat(input, seq(2,2,1));
    input = temp[seq(inSize[0]/2, inSize[1]/2, 0), inSize];
  }

  if (getCroppedWidth() > 0 && getCroppedHeight() > 0) {
    dim_t croppedSize = seq(getCroppedWidth(), getCroppedHeight(), inSize[2]);

    tensor<float, 3> temp(croppedSize);
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

  tensor<float, 3> output = zeros<float>(outSize);

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

}
