/*
 * RandomImageTransformation.h
 *
 *  Created on: Nov 30, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RANDOMIMAGETRANSFORMATION_H_
#define GAPPUTILS_ML_RANDOMIMAGETRANSFORMATION_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class RandomImageTransformation : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RandomImageTransformation)
  Property(Input, boost::shared_ptr<std::vector<float> >)
  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(XRange, std::vector<int>)
  Property(YRange, std::vector<int>)
  Property(Output, boost::shared_ptr<std::vector<float> >)

private:
  mutable RandomImageTransformation* data;

public:
  RandomImageTransformation();
  virtual ~RandomImageTransformation();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RANDOMIMAGETRANSFORMATION_H_ */
