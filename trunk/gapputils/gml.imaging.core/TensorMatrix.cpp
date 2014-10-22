/*
 * TensorMatrix.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: tombr
 */

#include "TensorMatrix.h"

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/math.hpp>
#include <tbblas/repeat.hpp>
#include <capputils/attributes/OrAttribute.h>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(TensorMatrix)

  ReflectableBase(DefaultWorkflowElement<TensorMatrix>)

  WorkflowProperty(InputTensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(IndividualMatrices, Flag())
  WorkflowProperty(MaxCount)
  WorkflowProperty(ColumnCount, Description("The number of columns. A value of -1 indicates to always use a squared matrix."))
  WorkflowProperty(TilingPlane, Enumerator<Type>())
  WorkflowProperty(TensorMatrix, Output("T"))
  WorkflowProperty(TensorMatrices, Output("Ts"))

EndPropertyDefinitions

TensorMatrix::TensorMatrix() : _IndividualMatrices(false), _MaxCount(-1), _ColumnCount(-1) {
  setLabel("Matrix");
}

void TensorMatrix::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor_t::dim_t dim_t;
  typedef tensor_t::value_t value_t;
  const int dimCount = tensor_t::dimCount;

  // anneliebttom copy this

  Logbook dlog = getLogbook();

  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getInputTensors();
  int count = getMaxCount() > 0 ? std::min(getMaxCount(), (int)inputs.size()) : inputs.size();

  if (getIndividualMatrices()) {
    boost::shared_ptr<v_tensor_t> outputs(new v_tensor_t());
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensor_t& input = *inputs[i];

      const int channelCount = input.size()[dimCount - 1];

      int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)channelCount));
      int rowCount = ceil((float)channelCount / (float)columnCount);

      dim_t inSize = input.size();
      dim_t channelSize = inSize;
      channelSize[dimCount - 1] = 1;
      dim_t outSize = channelSize;

      switch (getTilingPlane()) {
        case TilingPlane::Axial: outSize = outSize * seq(columnCount, rowCount, 1, 1); break;
        case TilingPlane::Sagittal: outSize = outSize * seq(1, columnCount, rowCount, 1); break;
        case TilingPlane::Coronal: outSize = outSize * seq(columnCount, 1, rowCount, 1); break;

        default:
          dlog(Severity::Warning) << "Unsupported tiling plane. Using axial";
          outSize = outSize * seq(columnCount, rowCount, 1, 1);
      }

      boost::shared_ptr<tensor_t> output(new tensor_t(zeros<value_t>(outSize)));
      for (int y = 0, z = 0; y < rowCount; ++y) {
        for (int x = 0; x < columnCount && z < channelCount; ++x, ++z) {
          switch (getTilingPlane()) {
          case TilingPlane::Sagittal:
            (*output)[seq(0, x, rowCount - y - 1, 0) * channelSize, channelSize] = input[seq(0, 0, 0, z), channelSize];
            break;

          case TilingPlane::Coronal:
            (*output)[seq(x, 0, rowCount - y - 1, 0) * channelSize, channelSize] = input[seq(0, 0, 0, z), channelSize];
            break;

          default:  // Axial is the default
            (*output)[seq(x, y, 0, 0) * channelSize, channelSize] = input[seq(0, 0, 0, z), channelSize];
          }
        }
      }

      outputs->push_back(output);
    }
    newState->setTensorMatrices(outputs);
  } else {
    int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)count));
    int rowCount = ceil((float)count / (float)columnCount);

    dim_t inSize = inputs[0]->size();
    dim_t outSize = inSize;

    switch (getTilingPlane()) {
      case TilingPlane::Axial: outSize = outSize * seq(columnCount, rowCount, 1, 1); break;
      case TilingPlane::Sagittal: outSize = outSize * seq(1, columnCount, rowCount, 1); break;
      case TilingPlane::Coronal: outSize = outSize * seq(columnCount, 1, rowCount, 1); break;

      default:
        dlog(Severity::Warning) << "Unsupported tiling plane. Using axial";
        outSize = outSize * seq(columnCount, rowCount, 1, 1);
    }

    boost::shared_ptr<tensor_t> output(new tensor_t(zeros<value_t>(outSize)));
    for (int y = 0, z = 0; y < rowCount; ++y) {
      for (int x = 0; x < columnCount && z < count; ++x, ++z) {
        switch (getTilingPlane()) {
        case TilingPlane::Sagittal:
          (*output)[seq(0, x, rowCount - y - 1, 0) * inSize, inSize] = *inputs[z];
          break;

        case TilingPlane::Coronal:
          (*output)[seq(x, 0, rowCount - y - 1, 0) * inSize, inSize] = *inputs[z];
          break;

        default:  // Axial is the default
          (*output)[seq(x, y, 0, 0) * inSize, inSize] = *inputs[z];
        }
      }
    }
    newState->setTensorMatrix(output);
  }
}

}

} /* namespace convrbm4d */

} /* namespace gml */
