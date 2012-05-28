/*
 * ImageCombiner.h
 *
 *  Created on: Jul 22, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_IMAGECOMBINER_H_
#define GAPPUTILSCV_IMAGECOMBINER_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Enumerators.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

ReflectableEnum(CombinerMode, Add, Subtract, Multiply, Divide);

class ImageCombiner : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ImageCombiner)

  Property(InputImage1, boost::shared_ptr<culib::ICudaImage>)
  Property(InputImage2, boost::shared_ptr<culib::ICudaImage>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Mode, CombinerMode)

private:
  mutable ImageCombiner* data;

public:
  ImageCombiner();
  virtual ~ImageCombiner();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_IMAGECOMBINER_H_ */
