/*
 * CrossValidateTensors.h
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#ifndef GML_CROSSVALIDATETENSORS_H_
#define GML_CROSSVALIDATETENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class CrossValidateTensors : public DefaultWorkflowElement<CrossValidateTensors> {

  InitReflectableClass(CrossValidateTensors)

  Property(Dataset, boost::shared_ptr<v_host_tensor_t>)
  Property(Interleaved, bool)
  Property(CurrentFold, int)
  Property(FoldCount, int)
  Property(TrainingCount, int)
  Property(TestCount, int)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(TestSet, boost::shared_ptr<v_host_tensor_t>)

public:
  CrossValidateTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_CROSSVALIDATETENSORS_H_ */
