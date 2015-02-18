/*
 * TestThreshold2.h
 *
 *  Created on: Feb 4, 2015
 *      Author: tombr
 */

#ifndef GML_TESTTHRESHOLD2_H_
#define GML_TESTTHRESHOLD2_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace encoder {

class TestThreshold2 : public DefaultWorkflowElement<TestThreshold2> {

  typedef std::vector<double> data_t;

  InitReflectableClass(TestThreshold2)

  Property(TrainingMaps, boost::shared_ptr<v_host_tensor_t>)
  Property(TrainingLabels, boost::shared_ptr<v_host_tensor_t>)
  Property(TestMaps, boost::shared_ptr<v_host_tensor_t>)
  Property(TestLabels, boost::shared_ptr<v_host_tensor_t>)
  Property(TrainTPR, boost::shared_ptr<data_t>)
  Property(TrainPPV, boost::shared_ptr<data_t>)
  Property(TrainDSC, boost::shared_ptr<data_t>)
  Property(TestTPR, boost::shared_ptr<data_t>)
  Property(TestPPV, boost::shared_ptr<data_t>)
  Property(TestDSC, boost::shared_ptr<data_t>)

public:
  TestThreshold2();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_TESTTHRESHOLD2_H_ */
