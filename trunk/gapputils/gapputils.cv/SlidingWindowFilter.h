/*
 * SlidingWindowFilter.h
 *
 *  Created on: May 30, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_SLIDINGWINDOWFILTER_H_
#define GAPPUTILS_CV_SLIDINGWINDOWFILTER_H_

#include <gapputils/WorkflowElement.h>

#include <gapputils/Image.h>

#include "AggregatorFunction.h"

namespace gapputils {

namespace cv {

class SlidingWindowFilter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(SlidingWindowFilter)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Filter, AggregatorFunction)
  Property(FilterSize, int)
  Property(OutputImage, boost::shared_ptr<image_t>)

private:
  mutable SlidingWindowFilter* data;

public:
  SlidingWindowFilter();
  virtual ~SlidingWindowFilter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_CV_SLIDINGWINDOWFILTER_H_ */
