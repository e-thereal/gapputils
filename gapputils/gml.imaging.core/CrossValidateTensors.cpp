/*
 * CrossValidateTensors.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#include "CrossValidateTensors.h"
#include <capputils/attributes/GreaterThanAttribute.h>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(CrossValidateTensors)

  ReflectableBase(DefaultWorkflowElement<CrossValidateTensors>)

  WorkflowProperty(Dataset, Input("Data"), NotEmpty<Type>())
  WorkflowProperty(Interleaved, Flag())
  WorkflowProperty(CurrentFold, Description("Zero-based index."))
  WorkflowProperty(FoldCount, Description("If 0, the entire data set will be copied to both, the training and the test set."))
  WorkflowProperty(TrainingCount, NoParameter())
  WorkflowProperty(TestCount, NoParameter())
  WorkflowProperty(TrainingSet, Output("Train"))
  WorkflowProperty(TestSet, Output("Test"))

EndPropertyDefinitions

CrossValidateTensors::CrossValidateTensors() : _Interleaved(false), _CurrentFold(0), _FoldCount(5), _TrainingCount(0), _TestCount(0) {
  setLabel("CrossVal");
}

void CrossValidateTensors::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  v_host_tensor_t& dataset = *getDataset();
  boost::shared_ptr<v_host_tensor_t> training(new v_host_tensor_t());
  boost::shared_ptr<v_host_tensor_t> testing(new v_host_tensor_t());

  if (_FoldCount > 0) {
    if (dataset.size() % _FoldCount != 0) {
      dlog(Severity::Warning) << "The number of samples must be divisible by the FoldCount. Aborting!";
      return;
    }

    if (_CurrentFold >= _FoldCount) {
      dlog(Severity::Warning) << "The CurrentFold must be smaller than the FoldCount. Aborting!";
      return;
    }
    const int foldSize = (dataset.size() + getFoldCount() - 1) / getFoldCount();
    for (int i = 0; i < (int)dataset.size(); ++i) {
      const int idx = _Interleaved ? i / foldSize + (i % foldSize) * _FoldCount : i;
      if (i >= getCurrentFold() * foldSize && i < (getCurrentFold() + 1) * foldSize)
        testing->push_back(dataset[idx]);
      else
        training->push_back(dataset[idx]);
    }
  } else {
    training->resize(dataset.size());
    testing->resize(dataset.size());

    std::copy(dataset.begin(), dataset.end(), training->begin());
    std::copy(dataset.begin(), dataset.end(), testing->begin());
  }

  newState->setTrainingCount(training->size());
  newState->setTestCount(testing->size());
  newState->setTrainingSet(training);
  newState->setTestSet(testing);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
