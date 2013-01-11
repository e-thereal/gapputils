/*
 * VolumeMatrix.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#include "VolumeMatrix.h"

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/math.hpp>
#include <tbblas/repeat.hpp>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(VolumeMatrix)

  ReflectableBase(DefaultWorkflowElement<VolumeMatrix>)

  WorkflowProperty(InputVolumes, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(MaxCount)
  WorkflowProperty(MinValue)
  WorkflowProperty(MaxValue)
  WorkflowProperty(AutoScale)
  WorkflowProperty(ColumnCount,
      Description("The number of columns. A value of -1 indicates to always use a squared matrix."))
  WorkflowProperty(CenterImages)
  WorkflowProperty(CroppedWidth)
  WorkflowProperty(CroppedHeight)
  WorkflowProperty(CroppedDepth)
  WorkflowProperty(VolumeMatrix, Output("Out"))

EndPropertyDefinitions

VolumeMatrix::VolumeMatrix() : _MaxCount(-1), _MinValue(0), _MaxValue(1), _AutoScale(false),
_ColumnCount(-1), _CenterImages(false), _CroppedWidth(-1), _CroppedHeight(-1), _CroppedDepth(-1)
{
  setLabel("Matrix");
}

void VolumeMatrix::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor<float, 4>::dim_t dim_t;

  // anneliebttom copy this

  Logbook dlog = getLogbook();

  std::vector<boost::shared_ptr<image_t> >& inputs = *getInputVolumes();

  dim_t inSize = seq(inputs[0]->getSize()[0], inputs[0]->getSize()[1], inputs[0]->getSize()[2],
      getMaxCount() > 0 ? std::min(getMaxCount(), (int)inputs.size()) : (unsigned)inputs.size());
  tensor<float, 4> input(inSize);

  for (int i = 0; i < inSize[3]; ++i)
    thrust::copy(inputs[i]->begin(), inputs[i]->end(), input.begin() + i * inputs[0]->getCount());

  int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)inSize[3]));
  int rowCount = ceil((float)inSize[3] / (float)columnCount);

  // centering
  if (getCenterImages()) {
    tensor<float, 4> temp = repeat(input, seq(2,2,2,1));
    input = temp[seq(inSize[0]/2, inSize[1]/2, inSize[2]/2, 0), inSize];
  }

  if (getCroppedWidth() > 0 && getCroppedHeight() > 0 && getCroppedDepth() > 0) {
    dim_t croppedSize = seq(getCroppedWidth(), getCroppedHeight(), getCroppedDepth(), inSize[3]);

    tensor<float, 4> temp(croppedSize);
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
  sliceSize[3] = 1;
  dim_t outSize = sliceSize * seq(columnCount, rowCount, 1, 1);

  tensor<float, 4> output = zeros<float>(outSize);

  for (int y = 0, z = 0; y < rowCount; ++y) {
    for (int x = 0; x < columnCount && z < inSize[3]; ++x, ++z) {
      output[seq(x, y, 0, 0) * sliceSize, sliceSize] = (input[seq(0, 0, 0, z), sliceSize] - minV) / (maxV - minV);
    }
  }

  boost::shared_ptr<image_t> outputImage(new image_t(outSize[0], outSize[1], outSize[2], inputs[0]->getPixelSize()));
  thrust::copy(output.begin(), output.end(), outputImage->begin());
  newState->setVolumeMatrix(outputImage);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
