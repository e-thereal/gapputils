/*
 * ImageCombiner.cpp
 *
 *  Created on: Jul 22, 2011
 *      Author: tombr
 */

#include "ImageCombiner.h"

#include <iostream>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ImageCombiner)

  ReflectableBase(DefaultWorkflowElement<ImageCombiner>)

  WorkflowProperty(InputImage1, Input("Img1"), NotNull<Type>())
  WorkflowProperty(InputImage2, Input("Img2"), NotNull<Type>())
  WorkflowProperty(Mode, Enumerator<CombinerMode>())
  WorkflowProperty(OutputImage, Output("Img"))

EndPropertyDefinitions

ImageCombiner::ImageCombiner() {
  setLabel("Combine");
}

void ImageCombiner::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& input1 = *getInputImage1();
  image_t& input2 = *getInputImage2();

  if (input1.getSize()[0] != input2.getSize()[0] ||
      input1.getSize()[1] != input2.getSize()[1] ||
      input1.getSize()[2] != input2.getSize()[2])
  {
    dlog(Severity::Warning) << "Dimensions don't match. Aborting!";
    return;
  }

  const int count = input1.getCount();

  boost::shared_ptr<image_t> output(new image_t(input1.getSize(), input1.getPixelSize()));

  float* buffer1 = input1.getData();
  float* buffer2 = input2.getData();
  float* buffer = output->getData();

  switch (getMode()) {
  case CombinerMode::Add:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] + buffer2[i];
    break;

  case CombinerMode::Subtract:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] - buffer2[i];
    break;

  case CombinerMode::Multiply:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] * buffer2[i];
    break;

  case CombinerMode::Divide:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] / buffer2[i];
    break;
  }

  newState->setOutputImage(output);
}

}

}
