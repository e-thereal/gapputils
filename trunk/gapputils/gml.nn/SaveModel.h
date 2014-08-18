/*
 * SaveModel.h
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef GML_SAVEMODEL_H_
#define GML_SAVEMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

class SaveModel : public DefaultWorkflowElement<SaveModel> {

  InitReflectableClass(SaveModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  SaveModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_SAVEMODEL_H_ */
