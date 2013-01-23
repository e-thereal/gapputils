#include "MergeChannels.h"

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(MergeChannels)

  ReflectableBase(DefaultWorkflowElement<MergeChannels>)

  WorkflowProperty(Channel1, Input("Ch1"), NotNull<Type>())
  WorkflowProperty(Channel2, Input("Ch2"), NotNull<Type>())
  WorkflowProperty(Channel3, Input("Ch3"), NotNull<Type>())
  WorkflowProperty(OutputImage, Output("Img"))

EndPropertyDefinitions

MergeChannels::MergeChannels() {
  setLabel("Merge");
}

void MergeChannels::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& channel1 = *getChannel1();
  image_t& channel2 = *getChannel2();
  image_t& channel3 = *getChannel3();
  const size_t width = channel1.getSize()[0];
  const size_t height = channel1.getSize()[1];
  const size_t depth = channel1.getSize()[2];
  const size_t slicePitch = width * height;

  if (width != channel2.getSize()[0] || height != channel2.getSize()[1] || depth != channel2.getSize()[2] ||
    width != channel2.getSize()[0] || height != channel2.getSize()[1] || depth != channel2.getSize()[2])
  {
    dlog(Severity::Warning) << "All channels must have the same dimension. Aborting!";
    return;
  }

  boost::shared_ptr<image_t> output(new image_t(width, height, 3 * depth, channel1.getPixelSize()));

  const size_t count = channel1.getCount();
  float *data = output->getData(), *data1 = channel1.getData(), *data2 = channel2.getData(), *data3 = channel3.getData();

  for (size_t i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    const size_t idx = i + 2 * (i / slicePitch) * slicePitch;

    data[idx] = data1[i];
    data[idx + slicePitch] = data2[i];
    data[idx + 2 * slicePitch] = data3[i];
    if (i % slicePitch == slicePitch - 1 && monitor)
      monitor->reportProgress(100.0 * i / count);
  }

  newState->setOutputImage(output);
}

}

}

}
