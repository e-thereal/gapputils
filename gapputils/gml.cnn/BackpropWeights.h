/*
 * BackpropWeights.h
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#ifndef GML_BACKPROPWEIGHTS_H_
#define GML_BACKPROPWEIGHTS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

struct BackpropWeightsChecker { BackpropWeightsChecker(); };

class BackpropWeights : public DefaultWorkflowElement<BackpropWeights> {

  friend class BackpropWeightsChecker;

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  InitReflectableClass(BackpropWeights)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Layer, int)
  Property(Weights, boost::shared_ptr<v_host_tensor_t>)

public:
  BackpropWeights();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cnn */

} /* namespace gml */

#endif /* GML_BACKPROPWEIGHTS_H_ */
