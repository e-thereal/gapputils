/*
 * RandomImageTransformation.h
 *
 *  Created on: Nov 30, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RANDOMIMAGETRANSFORMATION_H_
#define GAPPUTILS_ML_RANDOMIMAGETRANSFORMATION_H_

#include <gapputils/WorkflowElement.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace ml {

ReflectableEnum(TransformationType, Translation, Rotation, Scaling, Rigid);

class RandomImageTransformation : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RandomImageTransformation)
  Property(Input, boost::shared_ptr<std::vector<double> >)
  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Transformation, TransformationType)
  Property(XRange, std::vector<int>)
  Property(YRange, std::vector<int>)
  Property(ZRange, std::vector<int>)
  Property(Output, boost::shared_ptr<std::vector<double> >)

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
