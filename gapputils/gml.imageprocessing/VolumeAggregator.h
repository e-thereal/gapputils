/*
 * VolumeAggregator.h
 *
 *  Created on: Nov 19, 2013
 *      Author: tombr
 */

#ifndef GML_VOLUMEAGGREGATOR_H_
#define GML_VOLUMEAGGREGATOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include "AggregatorFunction.h"

namespace gml {

namespace imageprocessing {

class VolumeAggregator : public DefaultWorkflowElement<VolumeAggregator> {

  InitReflectableClass(VolumeAggregator)

  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Function, AggregatorFunction)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  VolumeAggregator();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_VOLUMEAGGREGATOR_H_ */
