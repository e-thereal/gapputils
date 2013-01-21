#pragma once
#ifndef GML_MERGECHANNELS_H_
#define GML_MERGECHANNELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class MergeChannels : public DefaultWorkflowElement<MergeChannels> {

  InitReflectableClass(MergeChannels)

  Property(Channel1, boost::shared_ptr<image_t>)
  Property(Channel2, boost::shared_ptr<image_t>)
  Property(Channel3, boost::shared_ptr<image_t>)

  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  MergeChannels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_MERGECHANNELS_H_ */