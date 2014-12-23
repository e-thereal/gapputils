/*
 * OpenPatchModel.h
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#ifndef GML_OPENPATCHMODEL_H_
#define GML_OPENPATCHMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

class OpenPatchModel : public DefaultWorkflowElement<OpenPatchModel> {

  InitReflectableClass(OpenPatchModel)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<patch_model_t>)
  Property(PatchSize, patch_model_t::dim_t)
  Property(HiddenCounts, std::vector<int>)
  Property(OutputCount, int)
  Property(LayerCount, int)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)

public:
  OpenPatchModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_OPENPATCHMODEL_H_ */
