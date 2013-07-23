/*
 * ModelWriter.h
 *
 *  Created on: Jul 09, 2013
 *      Author: tombr
 */

#ifndef GML_MODELWRITER_H_
#define GML_MODELWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbm {

class ModelWriter : public DefaultWorkflowElement<ModelWriter> {

  InitReflectableClass(ModelWriter)

  Property(Model, boost::shared_ptr<Model>)
  Property(Filename, std::string)
  Property(AutoSave, bool)
  Property(OutputName, std::string)

private:
  static int modelId;

public:
  ModelWriter();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_MODELWRITER_H_ */
