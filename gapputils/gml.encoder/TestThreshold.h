/*
 * TestThreshold.h
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#ifndef GML_TESTTHRESHOLD_H_
#define GML_TESTTHRESHOLD_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace encoder {

CapputilsEnumerator(Metric, Dsc, TprPpv, MinTprPpv);

struct TestThresholdChecker { TestThresholdChecker(); };

class TestThreshold : public DefaultWorkflowElement<TestThreshold> {

  typedef std::vector<double> data_t;

  static const int dimCount = host_tensor_t::dimCount;
  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;

  friend class TestThresholdChecker;

  InitReflectableClass(TestThreshold)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(TrainingLabels, boost::shared_ptr<v_host_tensor_t>)
  Property(TestSet, boost::shared_ptr<v_host_tensor_t>)
  Property(TestLabels, boost::shared_ptr<v_host_tensor_t>)
  Property(Metric, Metric)
  Property(FilterBatchSize, std::vector<int>)
  Property(SubRegionCount, host_tensor_t::dim_t)
  Property(GlobalTPR, boost::shared_ptr<data_t>)
  Property(GlobalPPV, boost::shared_ptr<data_t>)
  Property(GlobalDSC, boost::shared_ptr<data_t>)
  Property(OptimalTPR, boost::shared_ptr<data_t>)
  Property(OptimalPPV, boost::shared_ptr<data_t>)
  Property(OptimalDSC, boost::shared_ptr<data_t>)
  Property(PredictedTPR, boost::shared_ptr<data_t>)
  Property(PredictedPPV, boost::shared_ptr<data_t>)
  Property(PredictedDSC, boost::shared_ptr<data_t>)

public:
  TestThreshold();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_TESTTHRESHOLD_H_ */
