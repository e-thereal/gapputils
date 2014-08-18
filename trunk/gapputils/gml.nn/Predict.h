/*
 * Predict.h
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef GML_PREDICT_H_
#define GML_PREDICT_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

struct PredictChecker { PredictChecker(); };

class Predict : public DefaultWorkflowElement<Predict> {

  friend class PredictChecker;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Predict)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Predict();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_PREDICT_H_ */
