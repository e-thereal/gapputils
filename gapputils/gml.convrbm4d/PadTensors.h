/*
 * PadTensors.h
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#ifndef GML_PADTENSORS_H_
#define GML_PADTENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "CodingDirection.h"

#include "Model.h"

namespace gml {

namespace convrbm4d {

class PadTensors : public DefaultWorkflowElement<PadTensors> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(PadTensors)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(OutputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)

public:
  PadTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_PADTENSORS_H_ */
