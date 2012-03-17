/*
 * ImageSaver.h
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_IMAGESAVER_H_
#define GAPPUTILSCV_IMAGESAVER_H_

#include <gapputils/WorkflowElement.h>
#include <qimage.h>

namespace gapputils {

namespace cv {

class ImageSaver : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageSaver)

  Property(ImagePtr, boost::shared_ptr<QImage>)
  Property(ImageName, std::string)
  Property(AutoSave, bool)
  Property(AutoName, std::string)
  Property(AutoSuffix, std::string)

private:
  mutable ImageSaver* data;
  static int imageId;
  int imageNumber;

public:
  ImageSaver();
  virtual ~ImageSaver();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_IMAGESAVER_H_ */
