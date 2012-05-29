/*
 * Register.h
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_REGISTER_H_
#define GAPPUTILS_CV_REGISTER_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include <culib/math3d.h>

#include <capputils/Enumerators.h>

#include "SimilarityMeasure.h"

namespace gapputils {

namespace cv {

ReflectableEnum(OptimizerType, Powell, Simplex);

class Register : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Register)

  Property(BaseImage, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Similarity, SimilarityMeasure)
  Property(Optimizer, OptimizerType)
  Property(InPlane, bool)
  Property(Matrix, boost::shared_ptr<fmatrix4>)

private:
  mutable Register* data;

public:
  Register();
  virtual ~Register();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_REGISTER_H_ */
