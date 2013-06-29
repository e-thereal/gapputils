/*
 * CreateDbm.h
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#ifndef GML_CREATEDBM_H_
#define GML_CREATEDBM_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "DbmModel.h"

namespace gml {

namespace convrbm4d {

struct CreateDbmChecker { CreateDbmChecker(); };

class CreateDbm : public DefaultWorkflowElement<CreateDbm> {

  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;
  typedef Model::v_tensor_t v_host_tensor_t;
  typedef DbmModel::vv_tensor_t vv_host_tensor_t;

  friend class CreateDbmChecker;

  InitReflectableClass(CreateDbm)

  Property(Dataset, boost::shared_ptr<v_host_tensor_t>)
  Property(CrbmModels, boost::shared_ptr<std::vector<boost::shared_ptr<Model> > >)
  Property(DbmModel, boost::shared_ptr<DbmModel>)

public:
  CreateDbm();

protected:
  virtual void update(IProgressMonitor* monitor) const;

};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CREATEDBM_H_ */
