/*
 * OpenModel.h
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef GML_OPENMODEL_H_
#define GML_OPENMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace encoder {

class OpenModel : public DefaultWorkflowElement<OpenModel> {

  InitReflectableClass(OpenModel)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<model_t>)
  Property(InputSize, model_t::dim_t)
  Property(OutputSize, model_t::dim_t)
  Property(FilterCounts, std::vector<int>)
  Property(HiddenCounts, std::vector<int>)
  Property(LayerCount, int)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)

public:
  OpenModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_OPENMODEL_H_ */
