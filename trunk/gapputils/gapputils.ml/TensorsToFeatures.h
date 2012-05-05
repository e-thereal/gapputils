/*
 * TensorsToFeatures.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TENSORSTOFEATURES_H_
#define GAPPUTILS_ML_TENSORSTOFEATURES_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

class TensorsToFeatures : public gapputils::workflow::WorkflowElement {
  typedef ConvRbmModel::value_t value_t;
  typedef ConvRbmModel::tensor_t tensor_t;

  InitReflectableClass(TensorsToFeatures)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Features, boost::shared_ptr<std::vector<value_t> >)
  Property(Auto, bool)

private:
  mutable TensorsToFeatures* data;
  static int inputId;

public:
  TensorsToFeatures();
  virtual ~TensorsToFeatures();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_TENSORSTOFEATURES_H_ */
