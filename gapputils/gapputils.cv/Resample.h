/*
 * Resample.h
 *
 *  Created on: Aug 31, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RESAMPLE_H_
#define GAPPUTILSCV_RESAMPLE_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class Resample : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Resample)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(OutputImages, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(Width, int)
  Property(Height, int)

private:
  mutable Resample* data;

public:
  Resample();
  virtual ~Resample();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_RESAMPLE_H_ */
