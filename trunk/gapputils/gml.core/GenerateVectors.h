/*
 * GenerateVectors.h
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#ifndef GML_GENERATEVECTORS_H_
#define GML_GENERATEVECTORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class GenerateVectors : public DefaultWorkflowElement<GenerateVectors> {

  InitReflectableClass(GenerateVectors)

  Property(From, std::vector<double>)
  Property(StepCount, std::vector<int>)
  Property(To, std::vector<double>)
  Property(Order, std::vector<int>)
  Property(Vectors, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)

public:
  GenerateVectors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_GENERATEVECTORS_H_ */
