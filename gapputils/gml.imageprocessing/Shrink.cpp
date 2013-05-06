/*
 * Shrink.cpp
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#include "Shrink.h"

#include <cmath>
#include <limits>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Shrink)

  ReflectableBase(DefaultWorkflowElement<Shrink>)

  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(WidthFactor)
  WorkflowProperty(HeightFactor)
  WorkflowProperty(DepthFactor)
  WorkflowProperty(ShrinkingMethod, Enumerator<Type>())
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

Shrink::Shrink() : _WidthFactor(1), _HeightFactor(1), _DepthFactor(1) {
  setLabel("Shrink");
}

void Shrink::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getWidthFactor() <= 0 || getHeightFactor() <= 0 || getDepthFactor() <= 0) {
    dlog(Severity::Warning) << "The factors must be greater than 0. Aborting!";
    return;
  }

  image_t& input = *getInputImage();
  const int iwidth = input.getSize()[0];
  const int iheight = input.getSize()[1];
  const int idepth = input.getSize()[2];

  const int owidth = input.getSize()[0] / getWidthFactor();
  const int oheight = input.getSize()[1] / getHeightFactor();
  const int odepth = input.getSize()[2] / getDepthFactor();
  const int count = getWidthFactor() * getHeightFactor() * getDepthFactor();

  boost::shared_ptr<image_t> output(new image_t(owidth, oheight, odepth, input.getPixelSize()));

  float *inbuf = input.getData(), *outbuf = output->getData();

  switch (getShrinkingMethod()) {
  case ShrinkingMethod::Average:
    for (int z = 0, i = 0; z < idepth; z += getDepthFactor()) {
      for (int y = 0; y < iheight; y += getHeightFactor()) {
        for (int x = 0; x < iwidth; x += getWidthFactor(), ++i) {
          float result = 0;
          for (int dz = 0; dz < getDepthFactor(); ++dz) {
            for (int dy = 0; dy < getHeightFactor(); ++dy) {
              for (int dx = 0; dx < getWidthFactor(); ++dx) {
                const int ix = x + dx;
                const int iy = y + dy;
                const int iz = z + dz;
                result += inbuf[(iz * iheight + iy) * iwidth + ix];
              }
            }
          }
          outbuf[i] = result / count;
        }
      }
    }
    break;

  case ShrinkingMethod::Maximum:
    for (int z = 0, i = 0; z < idepth; z += getDepthFactor()) {
      for (int y = 0; y < iheight; y += getHeightFactor()) {
        for (int x = 0; x < iwidth; x += getWidthFactor(), ++i) {
          float result = std::numeric_limits<float>::min();
          for (int dz = 0; dz < getDepthFactor(); ++dz) {
            for (int dy = 0; dy < getHeightFactor(); ++dy) {
              for (int dx = 0; dx < getWidthFactor(); ++dx) {
                const int ix = x + dx;
                const int iy = y + dy;
                const int iz = z + dz;
                result = std::max(result, inbuf[(iz * iheight + iy) * iwidth + ix]);
              }
            }
          }
          outbuf[i] = result;
        }
      }
    }
    break;

  case ShrinkingMethod::Minimum:
    for (int z = 0, i = 0; z < idepth; z += getDepthFactor()) {
      for (int y = 0; y < iheight; y += getHeightFactor()) {
        for (int x = 0; x < iwidth; x += getWidthFactor(), ++i) {
          float result = std::numeric_limits<float>::max();
          for (int dz = 0; dz < getDepthFactor(); ++dz) {
            for (int dy = 0; dy < getHeightFactor(); ++dy) {
              for (int dx = 0; dx < getWidthFactor(); ++dx) {
                const int ix = x + dx;
                const int iy = y + dy;
                const int iz = z + dz;
                result = std::min(result, inbuf[(iz * iheight + iy) * iwidth + ix]);
              }
            }
          }
          outbuf[i] = result;
        }
      }
    }
    break;
  }

  newState->setOutputImage(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
