/*
 * StackVolumes.h
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#ifndef GMLSTACKVOLUMES_H_
#define GMLSTACKVOLUMES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class StackVolumes : public DefaultWorkflowElement<StackVolumes> {

  InitReflectableClass(StackVolumes)

  Property(Volumes1, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Volumes2, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Volume1, boost::shared_ptr<image_t>)
  Property(Volume2, boost::shared_ptr<image_t>)

  Property(Output, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)

public:
  StackVolumes();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_STACKVOLUMES_H_ */
