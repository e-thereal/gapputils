/*
 * CreateDbm.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "CreateDbm.h"

#include <capputils/attributes/MergeAttribute.h>

namespace gml {

namespace dbm {

BeginPropertyDefinitions(CreateDbm)

  ReflectableBase(DefaultWorkflowElement<CreateDbm>)

  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(CrbmModels, Input("Crbm"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(RbmModels, Input("Rbm"), Merge<Type>())
  WorkflowProperty(DbmModel, Output("Dbm"))

EndPropertyDefinitions

CreateDbm::CreateDbm() {
  setLabel("DBM");
}

CreateDbmChecker createDbmChecker;

} /* namespace dbm */

} /* namespace gml */
