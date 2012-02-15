/*
 * ImageRepeater.h
 *
 *  Created on: Feb 13, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_IMAGEREPEATER_H_
#define GAPPUTILS_CV_IMAGEREPEATER_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class ImageRepeater : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageRepeater)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Count, int)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable ImageRepeater* data;

public:
  ImageRepeater();
  virtual ~ImageRepeater();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_CV_IMAGEREPEATER_H_ */
