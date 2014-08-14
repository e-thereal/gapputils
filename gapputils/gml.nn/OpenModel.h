/*
 * OpenModel.h
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef GML_OPENMODEL_H_
#define GML_OPENMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

class OpenModel : public DefaultWorkflowElement<OpenModel> {

  InitReflectableClass(OpenModel)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<nn_layer_t>)
  Property(VisibleCount, int)
  Property(HiddenCount, int)
  Property(ActivationFunction, tbblas::deeplearn::activation_function)

public:
  OpenModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_OPENMODEL_H_ */
