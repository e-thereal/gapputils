/*
 * Transform.h
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_TRANSFORM_H_
#define GAPPUTILS_CV_TRANSFORM_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include <culib/math3d.h>

namespace gapputils {

namespace cv {

class Transform : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Transform)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Matrix, boost::shared_ptr<fmatrix4>)
  Property(Width, int)
  Property(Height, int)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable Transform* data;

public:
  Transform();
  virtual ~Transform();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_TRANSFORM_H_ */
