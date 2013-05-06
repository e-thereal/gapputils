/*
 * Extend.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: tombr
 */

#include "Extend.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Extend)

  ReflectableBase(DefaultWorkflowElement<Extend>)

  WorkflowProperty(Input, Input(""), NotNull<Type>())
  WorkflowProperty(WidthFactor)
  WorkflowProperty(HeightFactor)
  WorkflowProperty(DepthFactor)
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

Extend::Extend() : _WidthFactor(1), _HeightFactor(1), _DepthFactor(1) {
  setLabel("Extend");
}

void Extend:: update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getWidthFactor() <= 0 || getHeightFactor() <= 0 || getDepthFactor() <= 0) {
    dlog(Severity::Warning) << "The factors must be greater than 0. Aborting!";
    return;
  }

  image_t& input = *getInput();
  const int width = input.getSize()[0] * getWidthFactor();
  const int height = input.getSize()[1] * getHeightFactor();
  const int depth = input.getSize()[2] * getDepthFactor();

  boost::shared_ptr<image_t> output(new image_t(width, height, depth, input.getPixelSize()));

  float *inbuf = input.getData(), *outbuf = output->getData();

  for (int z = 0, i = 0; z < depth; z += getDepthFactor()) {
    for (int y = 0; y < height; y += getHeightFactor()) {
      for (int x = 0; x < width; x += getWidthFactor(), ++i) {
        for (int dz = 0; dz < getDepthFactor(); ++dz) {
          for (int dy = 0; dy < getHeightFactor(); ++dy) {
            for (int dx = 0; dx < getWidthFactor(); ++dx) {
              const int ox = x + dx;
              const int oy = y + dy;
              const int oz = z + dz;
              outbuf[(oz * height + oy) * width + ox] = inbuf[i];
            }
          }
        }
      }
    }
  }

  newState->setOutput(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
