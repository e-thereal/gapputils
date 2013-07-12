/*
 * DbmWriter.h
 *
 *  Created on: Jul 09, 2013
 *      Author: tombr
 */

#ifndef GML_DBMWRITER_H_
#define GML_DBMWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "DbmModel.h"

namespace gml {

namespace convrbm4d {

class DbmWriter : public DefaultWorkflowElement<DbmWriter> {

  InitReflectableClass(DbmWriter)

  Property(Model, boost::shared_ptr<DbmModel>)
  Property(Filename, std::string)
  Property(AutoSave, bool)
  Property(OutputName, std::string)

private:
  static int modelId;

public:
  DbmWriter();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* DBMWRITER_H_ */
