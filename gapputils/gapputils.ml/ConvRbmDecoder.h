/*
 * ConvRbmDecoder.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMDECODER_H_
#define GAPPUTILS_ML_CONVRBMDECODER_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

class ConvRbmDecoder : public gapputils::workflow::WorkflowElement {
public:
  typedef ConvRbmModel::tensor_t host_tensor_t;
  typedef ConvRbmModel::value_t value_t;
  typedef tbblas::tensor_base<value_t, 3, true> device_tensor_t;

  InitReflectableClass(ConvRbmDecoder)

  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(SampleVisibles, bool)
  Property(Auto, bool)

private:
  mutable ConvRbmDecoder* data;
  static int inputId;

public:
  ConvRbmDecoder();
  virtual ~ConvRbmDecoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMDECODER_H_ */
