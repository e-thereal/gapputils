/*
 * SplitModel.h
 *
 *  Created on: Dec 12, 2013
 *      Author: tombr
 */

#ifndef GML_SPLITMODEL_H_
#define GML_SPLITMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace rbm {

class SplitModel : public DefaultWorkflowElement<SplitModel> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Weights, boost::shared_ptr<v_data_t>)
  Property(VisibleBias, boost::shared_ptr<data_t>)
  Property(HiddenBias, boost::shared_ptr<data_t>)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace rbm */

} /* namespace gml */

#endif /* GML_SPLITMODEL_H_ */
