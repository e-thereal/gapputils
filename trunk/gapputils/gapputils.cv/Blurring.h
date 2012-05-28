/*
 * Blurring.h
 *
 *  Created on: May 27, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_BLURRING_H_
#define GAPPUTILS_CV_BLURRING_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include <culib/math3d.h>

namespace gapputils {

namespace cv {

class Blurring : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Blurring)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Sigma, float)
  Property(InPlane, bool)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable Blurring* data;

public:
  Blurring();
  virtual ~Blurring();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_REGISTER_H_ */
