#pragma once

#ifndef GAPPUTILSCV_FROMHSV_H_
#define GAPPUTILSCV_FROMHSV_H_

#include <gapputils/WorkflowElement.h>
#include <gapputils/Image.h>

#include <QImage>

namespace gapputils {

namespace cv {

class FromHsv : public workflow::WorkflowElement
{

  InitReflectableClass(FromHsv)

  Property(ImagePtr, boost::shared_ptr<QImage>)
  Property(Hue, boost::shared_ptr<image_t>)
  Property(Saturation, boost::shared_ptr<image_t>)
  Property(Value, boost::shared_ptr<image_t>)

private:
  mutable FromHsv* data;
  static int outputId;

public:
  FromHsv(void);
  virtual ~FromHsv(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_FROMHSV_H_ */
