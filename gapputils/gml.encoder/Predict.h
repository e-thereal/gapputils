/*
 * Predict.h
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#ifndef GML_PREDICT_H_
#define GML_PREDICT_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include <tbblas/deeplearn/objective_function.hpp>

#include "Model.h"

namespace gml {

namespace encoder {

struct PredictChecker { PredictChecker(); };

class Predict : public DefaultWorkflowElement<Predict> {

  friend class PredictChecker;

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Predict)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(MaximumLayer, int)
  Property(CalculateDeltas, bool)
  Property(Objective, tbblas::deeplearn::objective_function)
  Property(SensitivityRatio, double)
  Property(FilterBatchSize, std::vector<int>)
  Property(SubRegionCount, host_tensor_t::dim_t)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  Predict();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_PREDICT_H_ */
