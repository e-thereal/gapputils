/*
 * ConvertModel.cpp
 *
 *  Created on: Jun 27, 2014
 *      Author: tombr
 */

#include "ConvertModel.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(ConvertModel)

  ReflectableBase(DefaultWorkflowElement<ConvertModel>)

  WorkflowProperty(InputModel, Input(""), NotNull<Type>())
  WorkflowProperty(OutputModel, Output(""))

EndPropertyDefinitions

ConvertModel::ConvertModel() {
  setLabel("Convert");
}

void ConvertModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  boost::shared_ptr<model_t> crbm(new model_t());

  Model& model = *getInputModel();
  crbm->set_filters(*model.getFilters());
  crbm->set_visible_bias(*model.getVisibleBias());
  crbm->set_hidden_bias(*model.getHiddenBiases());
  crbm->set_mask(*model.getMask());
  crbm->set_kernel_size(model.getFilterKernelSize());
  unit_type unittype;
  unittype = model.getVisibleUnitType();
  crbm->set_visibles_type(unittype);
  unittype = model.getHiddenUnitType();
  crbm->set_hiddens_type(unittype);
  convolution_type convtype;
  convtype = model.getConvolutionType();
  crbm->set_convolution_type(convtype);
  crbm->set_mean(model.getMean());
  crbm->set_stddev(model.getStddev());

  newState->setOutputModel(crbm);
}

} /* namespace convrbm4d */

} /* namespace gml */
