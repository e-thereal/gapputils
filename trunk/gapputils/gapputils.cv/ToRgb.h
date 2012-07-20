#ifndef GAPPUTILS_TORGB_H
#define GAPPUTILS_TORGB_H

#include <gapputils/WorkflowElement.h>

#include <qimage.h>

#include <gapputils/Image.h>

namespace gapputils {

namespace cv {

class ToRgb : public workflow::WorkflowElement
{

InitReflectableClass(ToRgb)

Property(ImagePtr, boost::shared_ptr<QImage>)
Property(Red, boost::shared_ptr<image_t>)
Property(Green, boost::shared_ptr<image_t>)
Property(Blue, boost::shared_ptr<image_t>)

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
