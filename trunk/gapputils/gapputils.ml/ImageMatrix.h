/*
 * ImageMatrix.h
 *
 *  Created on: Dec 28, 2011
 *      Author: tombr
 */

#pragma once
#ifndef GAPPUTILS_CV_IMAGEMATRIX_H_
#define GAPPUTILS_CV_IMAGEMATRIX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <gapputils/Image.h>

namespace gapputils {

namespace ml {

struct ImageMatrixChecker { ImageMatrixChecker(); };

class ImageMatrix : public DefaultWorkflowElement<ImageMatrix> {

  friend class ImageMatrixChecker;

  InitReflectableClass(ImageMatrix)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(MinValue, float)
  Property(MaxValue, float)
  Property(ColumnCount, int)
  Property(ImageMatrix, boost::shared_ptr<image_t>)
  Property(AutoScale, bool)
  Property(CenterImages, bool)
  Property(CroppedWidth, int)
  Property(CroppedHeight, int)

private:
  static int inputId;

public:
  ImageMatrix();
  virtual ~ImageMatrix();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_ML_FGRBMWRITER_H_ */
