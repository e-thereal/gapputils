/*
 * RandomVector.h
 *
 *  Created on: Feb 27, 2013
 *      Author: tombr
 */

#ifndef GML_RANDOMVECTORS_H_
#define GML_RANDOMVECTORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {
namespace core {

class RandomVectors : public DefaultWorkflowElement<RandomVectors> {

  InitReflectableClass(RandomVectors)

  Property(Mean, double)
  Property(Stddev, double)
  Property(ElementCount, int)
  Property(VectorCount, int)
  Property(Vectors, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)

public:
  RandomVectors();

protected:
  void update(IProgressMonitor* monitor) const;
};

} /* namespace core */
} /* namespace gml */
#endif /* GML_RANDOMVECTORS_H_ */
