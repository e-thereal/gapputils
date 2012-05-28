/*
 * StackImages.h
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_STACKIMAGES_H_
#define GAPPUTILS_CV_STACKIMAGES_H_

#include <gapputils/WorkflowElement.h>
#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class StackImages : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(StackImages)

  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(InputImage1, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImage2, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImage3, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImage4, boost::shared_ptr<culib::ICudaImage>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable StackImages* data;

public:
  StackImages();
  virtual ~StackImages();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_CV_STACKIMAGES_H_ */
