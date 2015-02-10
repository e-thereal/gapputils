/*
 * CrossValidate.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#include "CrossValidate.h"
#include <capputils/attributes/GreaterThanAttribute.h>

namespace gml {

namespace core {

BeginPropertyDefinitions(CrossValidate)

  ReflectableBase(DefaultWorkflowElement<CrossValidate>)

  WorkflowProperty(Dataset, Input("Data"), NotEmpty<Type>())
  WorkflowProperty(Interleaved, Flag())
  WorkflowProperty(CurrentFold, Description("Zero-based index."))
  WorkflowProperty(FoldCount)
  WorkflowProperty(TrainingSet, Output("Train"))
  WorkflowProperty(TestSet, Output("Test"))

EndPropertyDefinitions

CrossValidate::CrossValidate() : _Interleaved(false), _CurrentFold(0), _FoldCount(5) {
  setLabel("CrossVal");
}

void CrossValidate::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  const std::vector<std::string>& dataset = getDataset();
  std::vector<std::string> training;
  std::vector<std::string> testing;

  if (dataset.size() % _FoldCount != 0) {
    dlog(Severity::Warning) << "The number of samples must be divisible by the FoldCount. Aborting!";
    return;
  }

  const int foldSize = (dataset.size() + getFoldCount() - 1) / getFoldCount();

  for (int i = 0; i < (int)dataset.size(); ++i) {
    const int idx = _Interleaved ? i / foldSize + (i % foldSize) * _FoldCount : i;
    if (i >= getCurrentFold() * foldSize && i < (getCurrentFold() + 1) * foldSize)
      testing.push_back(dataset[idx]);
    else
      training.push_back(dataset[idx]);
  }

  newState->setTrainingSet(training);
  newState->setTestSet(testing);
}

} /* namespace core */

} /* namespace gml */
