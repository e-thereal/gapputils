#pragma once
#ifndef GAPPUTILSCV_IMAGEWARP_H_
#define GAPPUTILSCV_IMAGEWARP_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include "GridModel.h"
#include <qimage.h>

namespace gapputils {

namespace cv {

class ImageWarp : public workflow::WorkflowElement
{

InitReflectableClass(ImageWarp)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(BaseGrid, boost::shared_ptr<GridModel>)
  Property(WarpedGrid, boost::shared_ptr<GridModel>)
  Property(Map, boost::shared_ptr<QImage>)

private:
  mutable ImageWarp* data;

public:
  ImageWarp(void);
  virtual ~ImageWarp(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_IMAGEWARP_H_ */
