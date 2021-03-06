/*
 * Predict.h
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#ifndef GML_PREDICT_H_
#define GML_PREDICT_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace jcnn {

struct PredictChecker { PredictChecker(); };

class Predict : public DefaultWorkflowElement<Predict> {

  friend class PredictChecker;

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  InitReflectableClass(Predict)

  Property(Model, boost::shared_ptr<model_t>)
  Property(LeftInputs, boost::shared_ptr<v_host_tensor_t>)
  Property(RightInputs, boost::shared_ptr<v_host_tensor_t>)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Predict();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace jcnn */

} /* namespace gml */

#endif /* GML_PREDICT_H_ */
