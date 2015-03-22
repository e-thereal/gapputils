/*
 * SaveModel.h
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef GML_SAVEMODEL_H_
#define GML_SAVEMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace encoder {

class SaveModel : public DefaultWorkflowElement<SaveModel> {

  InitReflectableClass(SaveModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Filename, std::string)
  Property(AutoSave, bool)
  Property(OutputName, std::string)

private:
  static int modelId;

public:
  SaveModel();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_SAVEMODEL_H_ */
