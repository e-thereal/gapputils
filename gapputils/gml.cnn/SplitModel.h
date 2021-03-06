/*
 * SplitModel.h
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef GML_SPLITMODEL_H_
#define GML_SPLITMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

class SplitModel : public DefaultWorkflowElement<SplitModel> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Layer, int)
  Property(Filters, boost::shared_ptr<v_tensor_t>)
  Property(Biases, boost::shared_ptr<v_tensor_t>)
  Property(Weights, boost::shared_ptr<v_data_t>)
  Property(Bias, boost::shared_ptr<data_t>)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cnn */

} /* namespace gml */

#endif /* GML_SPLITMODEL_H_ */
