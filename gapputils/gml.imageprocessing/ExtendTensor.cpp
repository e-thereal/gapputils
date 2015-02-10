/*
 * ExtendTensor.cpp
 *
 *  Created on: Feb 05, 2015
 *      Author: tombr
 */

#include "ExtendTensor.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ExtendTensor)

  ReflectableBase(DefaultWorkflowElement<ExtendTensor>)

  WorkflowProperty(Input, Input("T"))
  WorkflowProperty(Inputs, Input("Ts"))
  WorkflowProperty(WidthFactor)
  WorkflowProperty(HeightFactor)
  WorkflowProperty(DepthFactor)
  WorkflowProperty(Output, Output("T"))
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

ExtendTensor::ExtendTensor() : _WidthFactor(1), _HeightFactor(1), _DepthFactor(1) {
  setLabel("Extend");
}

void ExtendTensor:: update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getWidthFactor() <= 0 || getHeightFactor() <= 0 || getDepthFactor() <= 0) {
    dlog(Severity::Warning) << "The factors must be greater than 0. Aborting!";
    return;
  }

  if (getInput()) {
    host_tensor_t& input = *getInput();
    const int width = input.size()[0] * getWidthFactor();
    const int height = input.size()[1] * getHeightFactor();
    const int depth = input.size()[2] * getDepthFactor();
    const int channels = input.size()[3];

    boost::shared_ptr<host_tensor_t> output(new host_tensor_t(width, height, depth, channels));

    host_tensor_t::value_t *inbuf = input.data().data(), *outbuf = output->data().data();

    for (int c = 0, i = 0; c < channels; ++c) {
      for (int z = 0; z < depth; z += getDepthFactor()) {
        for (int y = 0; y < height; y += getHeightFactor()) {
          for (int x = 0; x < width; x += getWidthFactor(), ++i) {
            for (int dz = 0; dz < getDepthFactor(); ++dz) {
              for (int dy = 0; dy < getHeightFactor(); ++dy) {
                for (int dx = 0; dx < getWidthFactor(); ++dx) {
                  const int ox = x + dx;
                  const int oy = y + dy;
                  const int oz = z + dz;
                  outbuf[((c * depth + oz) * height + oy) * width + ox] = inbuf[i];
                }
              }
            }
          }
        }
      }
    }

    newState->setOutput(output);
  }

  if (getInputs()) {
    boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());
    v_host_tensor_t& inputs = *getInputs();

    for (size_t iTensor = 0; iTensor < inputs.size(); ++iTensor) {
      host_tensor_t& input = *inputs[iTensor];

      const int width = input.size()[0] * getWidthFactor();
      const int height = input.size()[1] * getHeightFactor();
      const int depth = input.size()[2] * getDepthFactor();
      const int channels = input.size()[3];

      boost::shared_ptr<host_tensor_t> output(new host_tensor_t(width, height, depth, channels));

      host_tensor_t::value_t *inbuf = input.data().data(), *outbuf = output->data().data();

      for (int c = 0, i = 0; c < channels; ++c) {
        for (int z = 0; z < depth; z += getDepthFactor()) {
          for (int y = 0; y < height; y += getHeightFactor()) {
            for (int x = 0; x < width; x += getWidthFactor(), ++i) {
              for (int dz = 0; dz < getDepthFactor(); ++dz) {
                for (int dy = 0; dy < getHeightFactor(); ++dy) {
                  for (int dx = 0; dx < getWidthFactor(); ++dx) {
                    const int ox = x + dx;
                    const int oy = y + dy;
                    const int oz = z + dz;
                    outbuf[((c * depth + oz) * height + oy) * width + ox] = inbuf[i];
                  }
                }
              }
            }
          }
        }
      }
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  }
}

} /* namespace imageprocessing */

} /* namespace gml */
