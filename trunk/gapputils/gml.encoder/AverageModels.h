/*
 * AverageModels.h
 *
 *  Created on: Jan 30, 2015
 *      Author: tombr
 */

#ifndef GML_AVERAGEMODELS_H_
#define GML_AVERAGEMODELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace encoder {

class AverageModels : public DefaultWorkflowElement<AverageModels> {

  InitReflectableClass(AverageModels)

  Property(Model1, boost::shared_ptr<model_t>)
  Property(Model2, boost::shared_ptr<model_t>)
  Property(AverageModel, boost::shared_ptr<model_t>)

public:
  AverageModels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_AVERAGEMODELS_H_ */
