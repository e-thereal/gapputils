/*
 * ConvRbmEncoder.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMENCODER_H_
#define GAPPUTILS_ML_CONVRBMENCODER_H_

#include <gapputils/WorkflowElement.h>
#include <capputils/Enumerators.h>

#include "ConvRbmModel.h"

#include <tbblas/device_tensor.hpp>

namespace gapputils {

namespace ml {

ReflectableEnum(CodingDirection, Encode, Decode);
ReflectableEnum(PoolingMethod, NoPooling, MaxPooling, PositionalMaxPooling, AvgPooling, StackPooling);

class ConvRbmEncoder : public gapputils::workflow::WorkflowElement {
public:
  typedef ConvRbmModel::tensor_t host_tensor_t;
  typedef ConvRbmModel::value_t value_t;
  typedef tbblas::device_tensor<value_t, 3> device_tensor_t;

  InitReflectableClass(ConvRbmEncoder)

  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(Sampling, bool)
  Property(Pooling, PoolingMethod)
  Property(Auto, bool)

private:
  mutable ConvRbmEncoder* data;
  static int inputId;

public:
  ConvRbmEncoder();
  virtual ~ConvRbmEncoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMENCODER_H_ */
