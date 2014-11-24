/*
 * Infer.h
 *
 *  Created on: Oct 28, 2014
 *      Author: tombr
 */

#ifndef GML_INFER_H_
#define GML_INFER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

struct InferChecker { InferChecker(); };

class Infer : public DefaultWorkflowElement<Infer> {

  typedef crbm_t::host_tensor_t host_tensor_t;
  typedef crbm_t::v_host_tensor_t v_host_tensor_t;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  friend class InferChecker;

  InitReflectableClass(Infer)

  Property(Model, boost::shared_ptr<dbn_t>)
  Property(InputTensors, boost::shared_ptr<v_host_tensor_t>)
  Property(InputUnits, boost::shared_ptr<v_data_t>)
  Property(Layer, int)
  Property(TopDown, bool)
  Property(GpuCount, int)
  Property(FilterBatchLength, std::vector<int>)
  Property(OutputTensors, boost::shared_ptr<v_host_tensor_t>)
  Property(OutputUnits, boost::shared_ptr<v_data_t>)

public:
  Infer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* GML_INFER_H_ */
