#include "SplitChannels.h"

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(SplitChannels)

  ReflectableBase(DefaultWorkflowElement<SplitChannels>)

  WorkflowProperty(InputImage, Input("Img"), NotNull<Type>())
  WorkflowProperty(Channel1, Output("Ch1"))
  WorkflowProperty(Channel2, Output("Ch2"))
  WorkflowProperty(Channel3, Output("Ch3"))

EndPropertyDefinitions

SplitChannels::SplitChannels() {
  setLabel("Split");
}

void SplitChannels::update(IProgressMonitor* monitor) const {

  image_t& input = *getInputImage();
  const size_t width = input.getSize()[0];
  const size_t height = input.getSize()[1];
  const size_t depth = input.getSize()[2];
  const size_t slicePitch = width * height;

  boost::shared_ptr<image_t> channel1(new image_t(width, height, depth / 3, input.getPixelSize()));
  boost::shared_ptr<image_t> channel2(new image_t(width, height, depth / 3, input.getPixelSize()));
  boost::shared_ptr<image_t> channel3(new image_t(width, height, depth / 3, input.getPixelSize()));

  const size_t count = channel1->getCount();
  float *data = input.getData(), *data1 = channel1->getData(), *data2 = channel2->getData(), *data3 = channel3->getData();

  for (size_t i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    const size_t idx = i + 2 * (i / slicePitch) * slicePitch;
    data1[i] = data[idx];
    data2[i] = data[idx + slicePitch];
    data3[i] = data[idx + 2 * slicePitch];
    if (i % slicePitch == slicePitch - 1 && monitor)
      monitor->reportProgress(100.0 * i / count);
  }

  newState->setChannel1(channel1);
  newState->setChannel2(channel2);
  newState->setChannel3(channel3);
}

}

}

}
