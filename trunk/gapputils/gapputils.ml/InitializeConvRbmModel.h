/*
 * InitializeConvRbmModel.h
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_INITIALIZECONVRBMMODEL_H_
#define GAPPUTILS_ML_INITIALIZECONVRBMMODEL_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

class InitializeConvRbmModel : public gapputils::workflow::WorkflowElement {
public:
  typedef ConvRbmModel::value_t value_t;
  typedef ConvRbmModel::tensor_t tensor_t;

private:
  InitReflectableClass(InitializeConvRbmModel)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(FilterCount, int)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(PoolingBlockSize, unsigned)
  Property(WeightMean, value_t)
  Property(WeightStddev, value_t)
  Property(IsGaussian, bool)

  Property(Model, boost::shared_ptr<ConvRbmModel>)

private:
  mutable InitializeConvRbmModel* data;

public:
  InitializeConvRbmModel();
  virtual ~InitializeConvRbmModel();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_INITIALIZECONVRBMMODEL_H_ */
