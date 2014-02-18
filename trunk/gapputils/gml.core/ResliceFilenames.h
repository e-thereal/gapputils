/*
 * ResliceFilenames.h
 *
 *  Created on: Feb 2, 2014
 *      Author: tombr
 */

#ifndef GML_RESLICEFILENAMES_H_
#define GML_RESLICEFILENAMES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class ResliceFilenames : public DefaultWorkflowElement<ResliceFilenames> {

  InitReflectableClass(ResliceFilenames)

  Property(Filenames, std::vector<std::string>)
  Property(Counts, std::vector<int>)
  Property(Order, std::vector<int>)
  Property(OutputNames, std::vector<std::string>)

public:
  ResliceFilenames();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_RESLICEFILENAMES_H_ */
