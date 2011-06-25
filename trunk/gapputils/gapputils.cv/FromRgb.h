#pragma once

#ifndef GAPPUTILSCV_FROMRGB_H_
#define GAPPUTILSCV_FROMRGB_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include <QImage>

namespace gapputils {

namespace cv {

class FromRgb : public workflow::WorkflowElement
{

  InitReflectableClass(FromRgb)

  Property(ImagePtr, QImage*)
  Property(Red, culib::ICudaImage*)
  Property(Green, culib::ICudaImage*)
  Property(Blue, culib::ICudaImage*)

private:
  mutable FromRgb* data;

public:
  FromRgb(void);
  virtual ~FromRgb(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_FROMRGB_H_ */
