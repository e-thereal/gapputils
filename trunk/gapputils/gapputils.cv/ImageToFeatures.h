/*
 * ImageToFeatures.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */
#ifndef GAPPUTILS_CV_IMAGETOFEATURES_H_
#define GAPPUTILS_CV_IMAGETOFEATURES_H_

#include <gapputils/WorkflowElement.h>
#include <boost/shared_ptr.hpp>
#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class ImageToFeatures : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageToFeatures)

  Property(Image, boost::shared_ptr<culib::ICudaImage>)
  Property(Data, boost::shared_ptr<std::vector<float> >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)

private:
  mutable ImageToFeatures* data;
  static int dataId;

public:
  ImageToFeatures();
  virtual ~ImageToFeatures();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_IMAGETOFEATURES_H_ */


