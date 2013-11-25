/*
 * Run.h
 *
 *  Created on: Nov 11, 2013
 *      Author: tombr
 */

#ifndef GML_RUN_H_
#define GML_RUN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Run : public DefaultWorkflowElement<Run> {

  InitReflectableClass(Run)

  Property(Input, std::string)
  Property(OutputName, std::string)
  Property(Command, std::string)
  Property(Output, std::string)

public:
  Run();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_RUN_H_ */
