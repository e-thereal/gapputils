#pragma once
#ifndef GAPPUTILSCV_TOHSV_H
#define GAPPUTILSCV_TOHSV_H

#include <gapputils/WorkflowElement.h>

#include <qimage.h>

#include <gapputils/Image.h>

namespace gapputils {

namespace cv {

class ToHsv : public workflow::WorkflowElement
{

InitReflectableClass(ToHsv)

Property(ImagePtr, boost::shared_ptr<QImage>)
Property(Hue, boost::shared_ptr<image_t>)
Property(Saturation, boost::shared_ptr<image_t>)
Property(Value, boost::shared_ptr<image_t>)

private:
  mutable ToHsv* data;

public:
  ToHsv(void);
  virtual ~ToHsv(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif // GAPPUTILSCV_TOHSV_H

