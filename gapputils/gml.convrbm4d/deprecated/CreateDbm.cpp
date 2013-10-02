/*
 * CreateDbm.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "CreateDbm.h"

#include <capputils/MergeAttribute.h>
#include <capputils/DeprecatedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(CreateDbm, Deprecated("Use gml.dbm.CreateDbm instead."))

  ReflectableBase(DefaultWorkflowElement<CreateDbm>)

  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(CrbmModels, Input("Crbm"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(DbmModel, Output("Dbm"))

EndPropertyDefinitions

CreateDbm::CreateDbm() {
  setLabel("DBM");
}

CreateDbmChecker createDbmChecker;

} /* namespace convrbm4d */

} /* namespace gml */
