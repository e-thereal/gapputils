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

namespace nn {

struct BackpropWeightsChecker { BackpropWeightsChecker(); };

class BackpropWeights : public DefaultWorkflowElement<BackpropWeights> {

  friend class BackpropWeightsChecker;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(BackpropWeights)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Layer, int)
  Property(Weights, boost::shared_ptr<v_data_t>)

public:
  BackpropWeights();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */
} /* namespace gml */
#endif /* BACKPROPWEIGHTS_H_ */
