#pragma once
#ifndef GAPPUTILSCV_IMAGEWARP_H_
#define GAPPUTILSCV_IMAGEWARP_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include "GridModel.h"

namespace gapputils {

namespace cv {

class ImageWarp : public workflow::WorkflowElement
{

InitReflectableClass(ImageWarp)

  Property(InputImage, culib::ICudaImage*)
  Property(OutputImage, culib::ICudaImage*)
  Property(BaseGrid, GridModel*)
  Property(WarpedGrid, GridModel*)

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