/*
 * FindThreshold.h
 *
 *  Created on: Jan 07, 2015
 *      Author: tombr
 */

#ifndef GML_FINDTHRESHOLD_H_
#define GML_FINDTHRESHOLD_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace encoder {

CapputilsEnumerator(Metric, Dsc, TprPpv, MinTprPpv);

struct FindThresholdChecker { FindThresholdChecker(); };

class FindThreshold : public DefaultWorkflowElement<FindThreshold> {

  typedef std::vector<double> data_t;

  static const int dimCount = host_tensor_t::dimCount;
  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;

  friend class FindThresholdChecker;

  InitReflectableClass(FindThreshold)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(Metric, Metric)
  Property(VoxelSize, host_tensor_t::dim_t)
  Property(TestThreshold, double)
//  Property(Model, boost::shared_ptr<model_t>)
  Property(LesionLoadsGlobal, boost::shared_ptr<data_t>)
  Property(LesionLoadsTest, boost::shared_ptr<data_t>)
  Property(LesionLoadsOptimal, boost::shared_ptr<data_t>)
  Property(LesionLoadsPredicted, boost::shared_ptr<data_t>)
  Property(PPV, boost::shared_ptr<data_t>)
  Property(TPR, boost::shared_ptr<data_t>)

public:
  FindThreshold();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_FINDTHRESHOLD_H_ */
