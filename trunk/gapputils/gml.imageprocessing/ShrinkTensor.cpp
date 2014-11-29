/*
 * ShrinkTensor.cpp
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#include "ShrinkTensor.h"

#include <cmath>
#include <limits>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ShrinkTensor)

  ReflectableBase(DefaultWorkflowElement<ShrinkTensor>)

  WorkflowProperty(InputTensor, Input(""), NotNull<Type>())
  WorkflowProperty(WidthFactor)
  WorkflowProperty(HeightFactor)
  WorkflowProperty(DepthFactor)
  WorkflowProperty(ShrinkingMethod, Enumerator<Type>())
  WorkflowProperty(OutputTensor, Output(""))

EndPropertyDefinitions

ShrinkTensor::ShrinkTensor() : _WidthFactor(1), _HeightFactor(1), _DepthFactor(1) {
  setLabel("Shrink");
}

void ShrinkTensor::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getWidthFactor() <= 0 || getHeightFactor() <= 0 || getDepthFactor() <= 0) {
    dlog(Severity::Warning) << "The factors must be greater than 0. Aborting!";
    return;
  }

  host_tensor_t& input = *getInputTensor();
  const int iwidth = input.size()[0];
  const int iheight = input.size()[1];
  const int idepth = input.size()[2];
  const int channels = input.size()[3];

  const int owidth = iwidth / getWidthFactor();
  const int oheight = iheight / getHeightFactor();
  const int odepth = idepth / getDepthFactor();

  boost::shared_ptr<host_tensor_t> output(new host_tensor_t(owidth, oheight, odepth, channels));

  float *inbuf = input.data().data(), *outbuf = output->data().data();

  switch (getShrinkingMethod()) {
  case ShrinkingMethod::Average:
    for (int c = 0, i = 0; c < channels; ++c) {
      for (int z = 0; z < odepth; ++z) {
        for (int y = 0; y < oheight; ++y) {
          for (int x = 0; x < owidth; ++x, ++i) {
            float result = 0;
            int count = 0;
            for (int dz = 0; dz < getDepthFactor(); ++dz) {
              for (int dy = 0; dy < getHeightFactor(); ++dy) {
                for (int dx = 0; dx < getWidthFactor(); ++dx) {
                  const int ix = x * getWidthFactor() + dx;
                  const int iy = y * getHeightFactor() + dy;
                  const int iz = z * getDepthFactor() + dz;
                  if (ix < iwidth && iy < iheight && iz < idepth) {
                    result += inbuf[((c * idepth + iz) * iheight + iy) * iwidth + ix];
                    ++count;
                  }
                }
              }
            }
            outbuf[i] = result / count;
          }
        }
      }
    }
    break;

//  case ShrinkingMethod::Maximum:
//    for (int z = 0, i = 0; z < odepth; ++z) {
//      for (int y = 0; y < oheight; ++y) {
//        for (int x = 0; x < owidth; ++x, ++i) {
//          float result = std::numeric_limits<float>::min();
//          for (int dz = 0; dz < getDepthFactor(); ++dz) {
//            for (int dy = 0; dy < getHeightFactor(); ++dy) {
//              for (int dx = 0; dx < getWidthFactor(); ++dx) {
//                const int ix = x * getWidthFactor() + dx;
//                const int iy = y * getHeightFactor() + dy;
//                const int iz = z * getDepthFactor() + dz;
//                if (ix < iwidth && iy < iheight && iz < idepth) {
//                  result = std::max(result, inbuf[(iz * iheight + iy) * iwidth + ix]);
//                }
//              }
//            }
//          }
//          outbuf[i] = result;
//        }
//      }
//    }
//    break;
//
//  case ShrinkingMethod::Minimum:
//    for (int z = 0, i = 0; z < odepth; ++z) {
//      for (int y = 0; y < oheight; ++y) {
//        for (int x = 0; x < owidth; ++x, ++i) {
//          float result = std::numeric_limits<float>::max();
//          for (int dz = 0; dz < getDepthFactor(); ++dz) {
//            for (int dy = 0; dy < getHeightFactor(); ++dy) {
//              for (int dx = 0; dx < getWidthFactor(); ++dx) {
//                const int ix = x * getWidthFactor() + dx;
//                const int iy = y * getHeightFactor() + dy;
//                const int iz = z * getDepthFactor() + dz;
//                if (ix < iwidth && iy < iheight && iz < idepth) {
//                  result = std::min(result, inbuf[(iz * iheight + iy) * iwidth + ix]);
//                }
//              }
//            }
//          }
//          outbuf[i] = result;
//        }
//      }
//    }
//    break;

  }

  newState->setOutputTensor(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
