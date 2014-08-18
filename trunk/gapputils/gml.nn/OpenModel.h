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
  Property(Model, boost::shared_ptr<model_t>)
  Property(InputCount, int)
  Property(HiddenCounts, std::vector<int>)
  Property(OutputCount, int)
  Property(LayerCount, int)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)

public:
  OpenModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_OPENMODEL_H_ */
