#pragma once
#ifndef GAPPUTILS_IMAGEINTERFACE_H_
#define GAPPUTILS_IMAGEINTERFACE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

namespace interfaces {

namespace inputs {

class Image : public gapputils::workflow::DefaultWorkflowElement<Image>
{
  InitReflectableClass(Image)

  Property(Value, boost::shared_ptr<gapputils::image_t>)

public:
  Image();
};

}

namespace outputs {

class Image : public gapputils::workflow::DefaultWorkflowElement<Image>
{
  InitReflectableClass(Image)

  Property(Value, boost::shared_ptr<gapputils::image_t>)

public:
  Image();
};

}

}

#endif
