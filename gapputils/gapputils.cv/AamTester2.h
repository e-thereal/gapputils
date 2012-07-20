/*
 * AamTester2.h
 *
 *  Created on: Jul 26, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMTESTER2_H_
#define GAPPUTILSCV_AAMTESTER2_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"
#include "GridModel.h"

namespace gapputils {

namespace cv {

class AamTester2 : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamTester2)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(Grid, boost::shared_ptr<GridModel>)
  Property(Image, boost::shared_ptr<image_t>)
  Property(AppearanceParameters, boost::shared_ptr<std::vector<float> >)
  Property(ShapeParameters, boost::shared_ptr<std::vector<float> >)
  Property(Similarity, double)

private:
  mutable AamTester2* data;

public:
  AamTester2();
  virtual ~AamTester2();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_AAMTESTER2_H_ */
