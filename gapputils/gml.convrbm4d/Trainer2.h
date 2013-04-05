/*
 * Trainer2.h
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#ifndef TRAINERTWO_H_
#define TRAINERTWO_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "SparsityMethod.h"

#include "Model.h"

namespace gml {
namespace convrbm4d {

struct Trainer2Checker { Trainer2Checker(); };

CapputilsEnumerator(FilterMethod, FFT, NaiveConvolution, OptimizedConvolution);

class Trainer2 : public DefaultWorkflowElement<Trainer2> {
public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;

  friend class Trainer2Checker;

  InitReflectableClass(Trainer2)

  Property(InitialModel, boost::shared_ptr<Model>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(GpuCount, int)
//  int dummy;
  Property(FilterMethod, FilterMethod)
  Property(LearningRateW, double)
  Property(LearningRateVB, double)
  Property(LearningRateHB, double)
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)
  Property(SparsityMethod, SparsityMethod)
  Property(RandomizeTraining, bool)
  Property(CalculateError, bool)
  Property(ShareBiasTerms, bool)
  Property(Logfile, std::string)
  Property(MonitorEvery, int)
  Property(ReconstructionCount, int)

  Property(Model, boost::shared_ptr<Model>)
  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(VisibleBiases, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(HiddenBiases, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(HiddenUnits, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Reconstructions, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(AverageEpochTime, double)

public:
  Trainer2();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* TRAINERTWO_H_ */
