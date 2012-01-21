/*
 * ImageToMif.h
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_CV_IMAGETOMIF_H_
#define GAPPUTLIS_CV_IMAGETOMIF_H_

#include <gapputils/WorkflowElement.h>
#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class ImageToMif : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageToMif)

  Property(Image, boost::shared_ptr<culib::ICudaImage>)
  Property(MinValue, double)
  Property(MaxValue, double)
  Property(AutoScale, bool)
  Property(MifName, std::string)

private:
  mutable ImageToMif* data;

public:
  ImageToMif();
  virtual ~ImageToMif();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTLIS_CV_IMAGETOMIF_H_ */
