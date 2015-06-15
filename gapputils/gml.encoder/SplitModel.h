/*
 * SplitModel.h
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef GML_SPLITMODEL_H_
#define GML_SPLITMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include "Model.h"

namespace gml {

namespace encoder {

class SplitModel : public DefaultWorkflowElement<SplitModel> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Layer, int)
  Property(Shortcut, bool)
  Property(Filters, boost::shared_ptr<v_host_tensor_t>)
  Property(Biases, boost::shared_ptr<v_host_tensor_t>)
  Property(Weights, boost::shared_ptr<v_data_t>)
  Property(Bias, boost::shared_ptr<data_t>)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_SPLITMODEL_H_ */
