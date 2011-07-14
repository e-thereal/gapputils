/*
 * Cropper.h
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_CROPPER_H_
#define GAPPUTILSCV_CROPPER_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

#include "RectangleModel.h"

namespace gapputils {

namespace cv {

class Cropper : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Cropper)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Rectangle, boost::shared_ptr<RectangleModel>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable Cropper* data;

public:
  Cropper();
  virtual ~Cropper();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_CROPPER_H_ */
