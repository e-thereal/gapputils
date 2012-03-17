/*
 * ImageMatrix.h
 *
 *  Created on: Dec 28, 2011
 *      Author: tombr
 */

#pragma once
#ifndef GAPPUTILS_CV_IMAGEMATRIX_H_
#define GAPPUTILS_CV_IMAGEMATRIX_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace ml {

class ImageMatrix : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageMatrix)
  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(MinValue, float)
  Property(MaxValue, float)
  Property(ImageMatrix, boost::shared_ptr<culib::ICudaImage>)
  Property(AutoScale, bool)
  Property(CenterImages, bool)

private:
  mutable ImageMatrix* data;
  static int inputId;

public:
  ImageMatrix();
  virtual ~ImageMatrix();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FGRBMWRITER_H_ */
