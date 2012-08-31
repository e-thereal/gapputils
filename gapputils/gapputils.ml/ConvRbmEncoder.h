/*
 * ConvRbmEncoder.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMENCODER_H_
#define GAPPUTILS_ML_CONVRBMENCODER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <capputils/Enumerators.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

CapputilsEnumerator(CodingDirection, Encode, Decode);
CapputilsEnumerator(PoolingMethod, NoPooling, MaxPooling, PositionalMaxPooling, AvgPooling, StackPooling);

class ConvRbmEncoder : public gapputils::workflow::DefaultWorkflowElement<ConvRbmEncoder> {
public:
  typedef ConvRbmModel::tensor_t host_tensor_t;
  typedef ConvRbmModel::value_t value_t;
  typedef tbblas::tensor_base<value_t, 3, true> device_tensor_t;

  InitReflectableClass(ConvRbmEncoder)

  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(Sampling, bool)
  Property(Pooling, PoolingMethod)
  Property(Auto, bool)
  Property(OutputDimension, std::vector<int>)

private:
  static int inputId;

public:
  ConvRbmEncoder();
  virtual ~ConvRbmEncoder();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMENCODER_H_ */
