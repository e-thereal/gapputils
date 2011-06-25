#ifndef GAPPUTILS_TORGB_H
#define GAPPUTILS_TORGB_H

#include <gapputils/WorkflowElement.h>

#include <qimage.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class ToRgb : public workflow::WorkflowElement
{

InitReflectableClass(ToRgb)

Property(ImagePtr, QImage*)
Property(Red, culib::ICudaImage*)
Property(Green, culib::ICudaImage*)
Property(Blue, culib::ICudaImage*)

private:
  mutable ToRgb* data;

public:
  ToRgb(void);
  virtual ~ToRgb(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif // GAPPUTILS_TORGB_H
