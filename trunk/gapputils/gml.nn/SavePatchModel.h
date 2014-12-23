/*
 * SavePatchModel.h
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#ifndef GML_SAVEPATCHMODEL_H_
#define GML_SAVEPATCHMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

class SavePatchModel : public DefaultWorkflowElement<SavePatchModel> {

  InitReflectableClass(SavePatchModel)

  Property(Model, boost::shared_ptr<patch_model_t>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  SavePatchModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_SAVEPATCHMODEL_H_ */
