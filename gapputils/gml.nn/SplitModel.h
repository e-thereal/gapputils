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

namespace nn {

class SplitModel : public DefaultWorkflowElement<SplitModel> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<nn_layer_t>)
  Property(Weights, boost::shared_ptr<v_data_t>)
  Property(Bias, boost::shared_ptr<data_t>)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_SPLITMODEL_H_ */
