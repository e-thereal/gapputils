/*
 * FeaturesToImage.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */
#ifndef GAPPUTILS_CV_FEATURESTOIMAGE_H_
#define GAPPUTILS_CV_FEATURESTOIMAGE_H_

#include <gapputils/WorkflowElement.h>
#include <boost/shared_ptr.hpp>
#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class FeaturesToImage : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FeaturesToImage)

  Property(Data, boost::shared_ptr<std::vector<float> >)
  Property(ColumnCount, int)
  Property(RowCount, int)
  Property(MaxCount, int)
  Property(Image, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable FeaturesToImage* data;
  static int dataId;

public:
  FeaturesToImage();
  virtual ~FeaturesToImage();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CVL_PRINCIPLECOMPONENTSTOMIF_H_ */


