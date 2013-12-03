/*
 * ApplyWarp.h
 *
 *  Created on: Nov 27, 2013
 *      Author: tombr
 */

#ifndef GML_APPLYWARP_H_
#define GML_APPLYWARP_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace fsl {

class ApplyWarp : public DefaultWorkflowElement<ApplyWarp> {

  InitReflectableClass(ApplyWarp)

  Property(Reference, std::string)
  Property(Input, std::string)
  Property(Warpfield, std::string)
  Property(OutputName, std::string)
  Property(ProgramName, std::string)
  Property(Output, std::string)

public:
  ApplyWarp();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace fsl */

} /* namespace gml */

#endif /* GML_APPLYWARP_H_ */
