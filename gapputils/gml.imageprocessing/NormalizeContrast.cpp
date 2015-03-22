/*
 * NormalizeContrast.cpp
 *
 *  Created on: Mar 1, 2013
 *      Author: tombr
 */

#include "NormalizeContrast.h"

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

#include <algorithm>

#include <capputils/EventHandler.h>

namespace gml {
namespace imageprocessing {

int NormalizeContrast::intensityId;

BeginPropertyDefinitions(NormalizeContrast)

  ReflectableBase(DefaultWorkflowElement<NormalizeContrast>)

  WorkflowProperty(InputImage, Input("I"))
  WorkflowProperty(InputImages, Input("Is"))
  WorkflowProperty(IntensityWindow, Dummy(intensityId = Id))
  WorkflowProperty(OutputImage, Output("I"))
  WorkflowProperty(OutputImages, Output("Is"))

EndPropertyDefinitions

NormalizeContrast::NormalizeContrast() : _IntensityWindow(2) {
  setLabel("Norm");
  _IntensityWindow[0] = 0;
  _IntensityWindow[1] = 1;

  Changed.connect(EventHandler<NormalizeContrast>(this, &NormalizeContrast::changedHandler));
}

void NormalizeContrast::changedHandler(ObservableClass* sender, int eventId) {
  if (eventId == intensityId && getIntensityWindow().size() != 2) {
    std::vector<double> window(2);
    window[0] = 0;
    window[1] = 1;
    setIntensityWindow(window);
  }
}

void NormalizeContrast::update(IProgressMonitor* monitor) const {
  using namespace tbblas;


  if (getInputImage()) {
    image_t& input = *getInputImage();
    boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

    tensor<float, 3> image1(input.getSize()[0], input.getSize()[1], input.getSize()[2]), image2;
    std::copy(input.begin(), input.end(), image1.begin());

    const float minimum = min(image1), maximum = max(image1);
  // Old equation:  image2 = max(0, image1) / max(image1);
    image2 = (image1 - minimum) / (maximum - minimum);

    const std::vector<double> window = getIntensityWindow();

    if (window[0] != 0.0 || window[1] != 1.0) {

      // Populate histogram
      std::vector<int> histogram(512);
      std::fill(histogram.begin(), histogram.end(), 0);
      for (tensor<float, 3>::iterator iter = image2.begin(); iter != image2.end(); ++iter)
        ++histogram[*iter * (histogram.size() - 1)];

      const int lowerSum = window[0] * image2.count(), upperSum = window[1] * image2.count();

      float lower = 0.0f, upper = 1.0f;
      for (int i = 0, sum = histogram[0]; i < (int)histogram.size() - 1; ++i) {
        if ((sum += histogram[i+1]) > lowerSum) {
          lower = (float)i / 511.f;
          break;
        }
      }

      for (int i = 0, sum = 0; i < (int)histogram.size(); ++i) {
        if ((sum += histogram[i]) >= upperSum) {
          upper = (float)i / 511.f;
          break;
        }
      }

      image1 = max(0, min(1.0, (image2 - lower) / (upper - lower)));
      std::copy(image1.begin(), image1.end(), output->begin());
    } else {
      std::copy(image2.begin(), image2.end(), output->begin());
    }

    newState->setOutputImage(output);
  }

  if (getInputImages()) {
    boost::shared_ptr<v_image_t> outputs(new v_image_t());
    for (size_t iImage = 0; iImage < _InputImages->size(); ++iImage) {
      image_t& input = *_InputImages->at(iImage);
      boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

      tensor<float, 3> image1(input.getSize()[0], input.getSize()[1], input.getSize()[2]), image2;
      std::copy(input.begin(), input.end(), image1.begin());

      const float minimum = min(image1), maximum = max(image1);
    // Old equation:  image2 = max(0, image1) / max(image1);
      image2 = (image1 - minimum) / (maximum - minimum);

      const std::vector<double> window = getIntensityWindow();

      if (window[0] != 0.0 || window[1] != 1.0) {

        // Populate histogram
        std::vector<int> histogram(512);
        std::fill(histogram.begin(), histogram.end(), 0);
        for (tensor<float, 3>::iterator iter = image2.begin(); iter != image2.end(); ++iter)
          ++histogram[*iter * (histogram.size() - 1)];

        const int lowerSum = window[0] * image2.count(), upperSum = window[1] * image2.count();

        float lower = 0.0f, upper = 1.0f;
        for (int i = 0, sum = histogram[0]; i < (int)histogram.size() - 1; ++i) {
          if ((sum += histogram[i+1]) > lowerSum) {
            lower = (float)i / 511.f;
            break;
          }
        }

        for (int i = 0, sum = 0; i < (int)histogram.size(); ++i) {
          if ((sum += histogram[i]) >= upperSum) {
            upper = (float)i / 511.f;
            break;
          }
        }

        image1 = max(0, min(1.0, (image2 - lower) / (upper - lower)));
        std::copy(image1.begin(), image1.end(), output->begin());
      } else {
        std::copy(image2.begin(), image2.end(), output->begin());
      }
      outputs->push_back(output);
    }
    newState->setOutputImages(outputs);
  }
}

} /* namespace imageprocessing */
} /* namespace gml */
