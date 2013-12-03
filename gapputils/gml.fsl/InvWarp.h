/*
 * InvWarp.h
 *
 *  Created on: Nov 27, 2013
 *      Author: tombr
 */

#ifndef GML_INVWARP_H_
#define GML_INVWARP_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace fsl {

class InvWarp : public DefaultWorkflowElement<InvWarp> {

  InitReflectableClass(InvWarp)

  Property(Reference, std::string)
  Property(Warpfield, std::string)
  Property(OutputName, std::string)
  Property(ProgramName, std::string)
  Property(InverseField, std::string)

public:
  InvWarp();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace gml */

}

#endif /* GML_INVWARP_H_ */
