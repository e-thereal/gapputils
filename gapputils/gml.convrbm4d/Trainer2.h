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

CapputilsEnumerator(FilterMethod, FFT, NaiveConvolution, OptimizedConvolution, ConvNet, NoConv);

class Trainer2 : public DefaultWorkflowElement<Trainer2> {
public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;

  friend class Trainer2Checker;

  InitReflectableClass(Trainer2)

  int dummy;

  Property(InitialModel, boost::shared_ptr<Model>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(GpuCount, int)
  Property(FilterMethod, FilterMethod)
  Property(Stride, int)
  int dummy2;
  Property(LearningRateW, double)
  Property(LearningRateVB, double)
  Property(LearningRateHB, double)
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)
  Property(SparsityMethod, SparsityMethod)
  Property(RandomizeTraining, bool)
  Property(CalculateError, bool)
  Property(ShareBiasTerms, bool)
//  Property(Logfile, std::string)
  int dummy3;

  Property(Model, boost::shared_ptr<Model>)
  Property(AverageEpochTime, double)

public:
  Trainer2();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* TRAINERTWO_H_ */
