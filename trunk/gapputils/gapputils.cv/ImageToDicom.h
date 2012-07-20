/*
 * ImageToDicom.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */
#ifndef GAPPUTILS_CV_IMAGETODICOM_H_
#define GAPPUTILS_CV_IMAGETODICOM_H_

#include <gapputils/WorkflowElement.h>
#include <gapputils/Image.h>

namespace gapputils {

namespace cv {

class ImageToDicom : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageToDicom)

  Property(Image, boost::shared_ptr<image_t>)
  Property(MinValue, double)
  Property(MaxValue, double)
  Property(AutoScale, bool)
  Property(Filename, std::string)

private:
  mutable ImageToDicom* data;

public:
  ImageToDicom();
  virtual ~ImageToDicom();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_CV_IMAGETODICOM_H_ */



