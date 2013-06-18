/*
 * Encoder.h
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#ifndef GML_ENCODER_H_
#define GML_ENCODER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "CodingDirection.h"
#include "Model.h"

namespace gml {

namespace dimreduce {

class Encoder : public DefaultWorkflowElement<Encoder> {

  InitReflectableClass(Encoder)

  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(Model, boost::shared_ptr<Model>)
  Property(Direction, CodingDirection)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)

public:
  Encoder();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dimreduce */
} /* namespace gml */
#endif /* ENCODER_H_ */
