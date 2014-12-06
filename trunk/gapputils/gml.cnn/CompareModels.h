/*
 * CompareModels.h
 *
 *  Created on: Dec 2, 2014
 *      Author: tombr
 */

#ifndef COMPAREMODELS_H_
#define COMPAREMODELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

class CompareModels : public DefaultWorkflowElement<CompareModels> {

  InitReflectableClass(CompareModels)

  Property(Model1, boost::shared_ptr<model_t>)
  Property(Model2, boost::shared_ptr<model_t>)

public:
  CompareModels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cnn */
} /* namespace gml */
#endif /* COMPAREMODELS_H_ */
