/*
 * ChangePooling.h
 *
 *  Created on: Aug 5, 2015
 *      Author: tombr
 */

#ifndef GML_CHANGEPOOLING_H_
#define GML_CHANGEPOOLING_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class ChangePooling : public DefaultWorkflowElement<ChangePooling> {

  InitReflectableClass(ChangePooling)

  Property(InputModel, boost::shared_ptr<model_t>)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(PoolingWidth, int)
  Property(PoolingHeight, int)
  Property(PoolingDepth, int)
  Property(OutputModel, boost::shared_ptr<model_t>)

public:
  ChangePooling();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CHANGEPOOLING_H_ */
