/*
 * ImageAggregator.h
 *
 *  Created on: May 18, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_IMAGEAGGREGATOR_H_
#define GAPPUTILS_CV_IMAGEAGGREGATOR_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

#include <capputils/Enumerators.h>

#include "AggregatorFunction.h"

namespace gapputils {

namespace cv {

class ImageAggregator : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageAggregator)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Function, AggregatorFunction)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable ImageAggregator* data;

public:
  ImageAggregator();
  virtual ~ImageAggregator();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_CV_IMAGEAGGREGATOR_H_ */
