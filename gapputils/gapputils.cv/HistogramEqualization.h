#pragma once
#ifndef GAPPUTILSCV_HISTOGRAMEQUALIZATION_H_
#define GAPPUTILSCV_HISTOGRAMEQUALIZATION_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class HistogramEqualization : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(HistogramEqualization)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(BinCount, int)

private:
  mutable HistogramEqualization* data;

public:
  HistogramEqualization();
  virtual ~HistogramEqualization();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_HISTOGRAMEQUALIZATION_H_ */
