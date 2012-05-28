/*
 * Transformation.h
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_TRANSFORMATION_H_
#define GAPPUTILS_CV_TRANSFORMATION_H_

#include <gapputils/WorkflowElement.h>

#include <culib/math3d.h>

namespace gapputils {

namespace cv {

class Transformation : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Transformation)

  Property(XTrans, float)
  Property(YTrans, float)
  Property(Matrix, boost::shared_ptr<fmatrix4>)

private:
  mutable Transformation* data;

public:
  Transformation();
  virtual ~Transformation();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_TRANSFORMATION_H_ */
