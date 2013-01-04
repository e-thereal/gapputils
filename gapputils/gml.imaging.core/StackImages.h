/*
 * StackImages.h
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#ifndef STACKIMAGES_H_
#define STACKIMAGES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Image.h>

namespace gml {

namespace imaging {

namespace core {

class StackImages : public DefaultWorkflowElement<StackImages> {

  InitReflectableClass(StackImages)

  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(InputImage1, boost::shared_ptr<image_t>)
  Property(InputImage2, boost::shared_ptr<image_t>)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  StackImages();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* STACKIMAGES_H_ */
