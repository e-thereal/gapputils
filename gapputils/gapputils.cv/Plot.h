/*
 * Plot.h
 *
 *  Created on: June 03, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_PLOT_H_
#define GAPPUTILS_CV_PLOT_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

class Plot : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Plot)

  Property(X, boost::shared_ptr<std::vector<float> >)
  Property(Y, boost::shared_ptr<std::vector<float> >)
  Property(Format, std::string)
  Property(Plot, std::string)

private:
  mutable Plot* data;

public:
  Plot();
  virtual ~Plot();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_PLOT_H_ */
