#pragma once
#ifndef GML_SPLITCHANNELS_H_
#define GML_SPLITCHANNELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class SplitChannels : public DefaultWorkflowElement<SplitChannels> {

  InitReflectableClass(SplitChannels)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Channel1, boost::shared_ptr<image_t>)
  Property(Channel2, boost::shared_ptr<image_t>)
  Property(Channel3, boost::shared_ptr<image_t>)

public:
  SplitChannels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif