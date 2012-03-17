/*
 * FeaturesToTensors.h
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FEATURESTOTENSORS_H_
#define GAPPUTILS_ML_FEATURESTOTENSORS_H_

#include "ConvRbmModel.h"

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class FeaturesToTensors : public gapputils::workflow::WorkflowElement {
  typedef ConvRbmModel::value_t value_t;
  typedef ConvRbmModel::tensor_t tensor_t;

  InitReflectableClass(FeaturesToTensors)

  Property(Features, boost::shared_ptr<std::vector<value_t> >)
  Property(Width, int)
  Property(Height, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)

private:
  mutable FeaturesToTensors* data;

public:
  FeaturesToTensors();
  virtual ~FeaturesToTensors();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FEATURESTOTENSORS_H_ */
