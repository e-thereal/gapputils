/*
 * CollectedFiles.h
 *
 *  Created on: May 6, 2013
 *      Author: tombr
 */

#ifndef GMLCOLLECTEDFILES_H_
#define GMLCOLLECTEDFILES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class CollectedFiles : public DefaultWorkflowElement<CollectedFiles> {

  InitReflectableClass(CollectedFiles)

  Property(Filenames, std::vector<std::string>)
  Property(Output, std::vector<std::string>)

public:
  CollectedFiles();

  virtual void show();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GMLCOLLECTEDFILES_H_ */
