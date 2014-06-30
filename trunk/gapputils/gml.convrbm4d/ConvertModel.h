/*
 * ConvertModel.h
 *
 *  Created on: Jun 27, 2014
 *      Author: tombr
 */

#ifndef GML_CONVERTMODEL_H_
#define GML_CONVERTMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class ConvertModel : public DefaultWorkflowElement<ConvertModel> {

  InitReflectableClass(ConvertModel)

  Property(InputModel, boost::shared_ptr<Model>)
  Property(OutputModel, boost::shared_ptr<model_t>)

public:
  ConvertModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* CONVERTMODEL_H_ */
