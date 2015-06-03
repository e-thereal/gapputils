/*
 * ModifyModel.h
 *
 *  Created on: Apr 10, 2015
 *      Author: tombr
 */

#ifndef GML_MODIFYMODEL_H_
#define GML_MODIFYMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace encoder {

class ModifyModel : public DefaultWorkflowElement<ModifyModel> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(ModifyModel)

  Property(InputModel, boost::shared_ptr<model_t>)
  Property(Filters, boost::shared_ptr<v_tensor_t>)
  Property(Biases, boost::shared_ptr<v_tensor_t>)
  Property(Weights, boost::shared_ptr<v_data_t>)
  Property(Bias, boost::shared_ptr<data_t>)
  Property(Layer, int)
  Property(Shortcut, bool)

  Property(OutputModel, boost::shared_ptr<model_t>)

public:
  ModifyModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_MODIFYMODEL_H_ */
