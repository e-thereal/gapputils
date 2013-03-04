/*
 * Specificity.h
 *
 *  Created on: Feb 28, 2013
 *      Author: tombr
 */

#ifndef GML_SPECIFICITY_H_
#define GML_SPECIFICITY_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

class Specificity : public DefaultWorkflowElement<Specificity> {
  typedef Model::tensor_t tensor_t;

  InitReflectableClass(Specificity)

  Property(GeneratedTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Dataset, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Minimum, double)
  Property(Maximum, double)
  Property(AverageRmse, double)

public:
  Specificity();

protected:
  void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* SPECIFICITY_H_ */
