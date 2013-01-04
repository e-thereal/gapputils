/*
 * ImageMatrix.h
 *
 *  Created on: Dec 28, 2011
 *      Author: tombr
 */

#pragma once
#ifndef IMAGEMATRIX_H_
#define IMAGEMATRIX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <gapputils/Image.h>

namespace gml {

namespace imaging {

namespace core {

class ImageMatrix : public DefaultWorkflowElement<ImageMatrix> {

  InitReflectableClass(ImageMatrix)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(ImageMatrix, boost::shared_ptr<image_t>)
  Property(MaxSliceCount, int)
  Property(MinValue, double)
  Property(MaxValue, double)
  Property(AutoScale, bool)
  Property(ColumnCount, int)
  Property(CenterImages, bool)
  Property(CroppedWidth, int)
  Property(CroppedHeight, int)

private:
  static int inputId;

public:
  ImageMatrix();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif
