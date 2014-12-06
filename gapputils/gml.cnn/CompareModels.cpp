/*
 * CompareModels.cpp
 *
 *  Created on: Dec 2, 2014
 *      Author: tombr
 */

#include "CompareModels.h"

#include <tbblas/deeplearn/serialize_cnn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/dot.hpp>

#include <sstream>

namespace gml {
namespace cnn {

BeginPropertyDefinitions(CompareModels)

  ReflectableBase(DefaultWorkflowElement<CompareModels>)

  WorkflowProperty(Model1, Input("M1"), NotNull<Type>())
  WorkflowProperty(Model2, Input("M2"), NotNull<Type>())

EndPropertyDefinitions

CompareModels::CompareModels() {
  setLabel("Compare");
}

void CompareModels::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  model_t &m1 = *getModel1(), &m2 = *getModel2();
  cnn_layer_t &l1 = *m1.cnn_layers()[0], &l2 = *m2.cnn_layers()[0];

  tbblas_print(l1.version() - l2.version());
  tbblas_print(l1.filter_count() - l2.filter_count());
  tbblas_print(l1.bias().size() - l2.bias().size());

  for (size_t i = 0; i < l1.filter_count(); ++i) {
    tbblas_print(dot(*l1.filters()[i] - *l2.filters()[i], *l1.filters()[i] - *l2.filters()[i]));
    tbblas_print(dot(*l1.bias()[i] - *l2.bias()[i], *l1.bias()[i] - *l2.bias()[i]));
  }

  tbblas_print(l1.kernel_size() - l2.kernel_size());
  tbblas_print(l1.stride_size() - l2.stride_size());
  tbblas_print(l1.pooling_size() - l2.pooling_size());
  tbblas_print(l1.mean() - l2.mean());
  tbblas_print(l1.stddev() - l2.stddev());
  tbblas_print(l1.shared_bias() - l2.shared_bias());

  tbblas_print(l1.activation_function());
  tbblas_print(l2.activation_function());
  tbblas_print(l1.convolution_type());
  tbblas_print(l2.convolution_type());
  tbblas_print(l1.pooling_method());
  tbblas_print(l2.pooling_method());
}

} /* namespace cnn */

} /* namespace gml */
