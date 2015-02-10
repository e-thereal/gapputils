/*
 * ImageAggregator.h
 *
 *  Created on: May 18, 2012
 *      Author: tombr
 */

#ifndef GML_IMAGEAGGREGATOR_H_
#define GML_IMAGEAGGREGATOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include "AggregatorFunction.h"

namespace gml {

namespace imageprocessing {

class ImageAggregator : public DefaultWorkflowElement<ImageAggregator> {

  InitReflectableClass(ImageAggregator)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Function, AggregatorFunction)
  Property(ScalarMode, bool)
  Property(OutputImage, boost::shared_ptr<image_t>)
  Property(OutputValue, double)

public:
  ImageAggregator();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_CV_IMAGEAGGREGATOR_H_ */
