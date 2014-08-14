/*
 * ReconstructionTest.h
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#ifndef GML_RECONSTRUCTIONTEST_H_
#define GML_RECONSTRUCTIONTEST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace dbn {

CapputilsEnumerator(TestType, Reconstruct, CalculateMSE, CalculateRMSE, CalculateRRMSE);

struct ReconstructionTestChecker { ReconstructionTestChecker(); };

class ReconstructionTest : public DefaultWorkflowElement<ReconstructionTest> {

  typedef crbm_t::host_tensor_t host_tensor_t;
  typedef crbm_t::v_host_tensor_t v_host_tensor_t;

  friend class ReconstructionTestChecker;

  InitReflectableClass(ReconstructionTest)

  Property(Model, boost::shared_ptr<dbn_t>)
  Property(Dataset, boost::shared_ptr<v_host_tensor_t>)
  Property(Type, TestType)
  Property(MaxLayer, int)
  Property(GpuCount, int)
  Property(FilterBatchLength, std::vector<int>)
  Property(Reconstructions, boost::shared_ptr<v_host_tensor_t>)
  Property(ReconstructionError, double)

public:
  ReconstructionTest();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* GML_RECONSTRUCTIONTEST_H_ */
