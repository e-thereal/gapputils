/*
 * Pooling.h
 *
 *  Created on: Apr 19, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_POOLING_H_
#define GAPPUTILS_CV_POOLING_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Enumerators.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

ReflectableEnum(PoolingDirection, Encode, Decode);

class Pooling : public gapputils::workflow::WorkflowElement {
  typedef ConvRbmModel::value_t value_t;
  typedef ConvRbmModel::tensor_t tensor_t;

  InitReflectableClass(Pooling)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(OutputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Direction, PoolingDirection)
  Property(Auto, bool)

private:
  mutable Pooling* data;
  static int inputId;

public:
  Pooling();
  virtual ~Pooling();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_POOLING_H_ */
