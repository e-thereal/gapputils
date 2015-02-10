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
  Logbook& dlog = getLogbook();

  boost::shared_ptr<dbn_t> dbn(new dbn_t());

  if (getCrbmModels())
    dbn->set_crbms(*getCrbmModels());
  if (getRbmModels())
    dbn->set_rbms(*getRbmModels());

  for (int i = 0; i < (int)dbn->crbms().size() - 1; ++i) {
    if (dbn->crbms()[i]->outputs_count() != dbn->crbms()[i + 1]->visibles_count()) {
      dlog(Severity::Warning) << "Number of output units of convRBM " << i << " not equal to number of visibles units of convRBM " << i + 1 << ". Aborting!";
      return;
    }
  }

  if (dbn->rbms().size() && dbn->crbms().size() &&
      dbn->crbms()[dbn->crbms().size() - 1]->outputs_count() != dbn->rbms()[0]->visibles_count())
  {
    dlog(Severity::Warning) << "Number of hidden units of convRBM " << dbn->crbms().size() - 1 << " not equal to number of visibles units of the first RBM. Aborting!";
    return;
  }

  for (int i = 0; i < (int)dbn->rbms().size() - 1; ++i) {
    if (dbn->rbms()[i]->hiddens_count() != dbn->rbms()[i + 1]->visibles_count()) {
      dlog(Severity::Warning) << "Number of hidden units of RBM " << i << " not equal to number of visibles units of RBM " << i + 1 << ". Aborting!";
      return;
    }
  }

  newState->setDbnModel(dbn);
}

} /* namespace dbn */

} /* namespace gml */
