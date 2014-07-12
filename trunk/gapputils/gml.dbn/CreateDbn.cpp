/*
 * CreateDbn.cpp
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#include "CreateDbn.h"

#include <capputils/attributes/MergeAttribute.h>

namespace gml {

namespace dbn {

BeginPropertyDefinitions(CreateDbn)

  ReflectableBase(DefaultWorkflowElement<CreateDbn>)

  WorkflowProperty(CrbmModels, Input("Crbm"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(RbmModels, Input("Rbm"), Merge<Type>())
  WorkflowProperty(DbnModel, Output("Dbn"))

EndPropertyDefinitions

CreateDbn::CreateDbn() {
  setLabel("DBN");
}

void CreateDbn::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<dbn_t> dbn(new dbn_t());

  if (getCrbmModels())
    dbn->set_crbms(*getCrbmModels());
  if (getRbmModels())
    dbn->set_rbms(*getRbmModels());

  newState->setDbnModel(dbn);
}

} /* namespace dbn */

} /* namespace gml */
