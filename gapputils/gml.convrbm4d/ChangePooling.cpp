/*
 * ChangePooling.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: tombr
 */

#include "ChangePooling.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(ChangePooling, Description("Changes the pooling information of a CRBM."))

  ReflectableBase(DefaultWorkflowElement<ChangePooling>)

  WorkflowProperty(InputModel, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(PoolingMethod, Enumerator<Type>())
  WorkflowProperty(PoolingWidth)
  WorkflowProperty(PoolingHeight)
  WorkflowProperty(PoolingDepth)
  WorkflowProperty(OutputModel, Output("CRBM"))

EndPropertyDefinitions

ChangePooling::ChangePooling() : _PoolingWidth(1), _PoolingHeight(1), _PoolingDepth(1) {
  setLabel("Pooling");
}

void ChangePooling::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<model_t> output(new model_t(*getInputModel()));
  output->set_pooling_method(getPoolingMethod());
  output->set_pooling_size(tbblas::seq(getPoolingWidth(), getPoolingHeight(), getPoolingDepth(), 1));
  newState->setOutputModel(output);
}

} /* namespace convrbm4d */

} /* namespace gml */
