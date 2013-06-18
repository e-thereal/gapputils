/*
 * Trainer.h
 *
 *  Created on: Jun 14, 2013
 *      Author: tombr
 */

#ifndef GML_TRAINER_H_
#define GML_TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "DimensionalityReductionMethod.h"
#include "Model.h"

namespace gml {

namespace dimreduce {

class Trainer : public DefaultWorkflowElement<Trainer> {

  InitReflectableClass(Trainer)

  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(Method, DimensionalityReductionMethod)
  Property(OutputDimension, int)
  Property(Neighbors, int)
  Property(Model, boost::shared_ptr<Model>)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)

public:
  Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;

};

} /* namespace gml */

} /* namespace gml */

#endif /* GML_TRAINER_H_ */
