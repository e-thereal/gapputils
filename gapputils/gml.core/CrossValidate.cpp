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

  WorkflowProperty(Files, Input("Files"))
  WorkflowProperty(Strings, Input("Strs"))
  WorkflowProperty(Interleaved, Flag())
  WorkflowProperty(CurrentFold, Description("Zero-based index."))
  WorkflowProperty(FoldCount, Description("If 0, the entire data set will be copied to both, the training and the test set."))
  WorkflowProperty(TrainingFiles, Output("TrainFs"))
  WorkflowProperty(TestFiles, Output("TestFs"))
  WorkflowProperty(TrainingStrings, Output("TrainSs"))
  WorkflowProperty(TestStrings, Output("TestSs"))

EndPropertyDefinitions

CrossValidate::CrossValidate() : _Interleaved(false), _CurrentFold(0), _FoldCount(5) {
  setLabel("CrossVal");
}

void CrossValidate::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (_Files.size()) {
    const std::vector<std::string>& dataset = _Files;
    std::vector<std::string> training;
    std::vector<std::string> testing;

    if (_FoldCount > 0) {

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
    } else {
      training.resize(dataset.size());
      testing.resize(dataset.size());

      std::copy(dataset.begin(), dataset.end(), training.begin());
      std::copy(dataset.begin(), dataset.end(), testing.begin());
    }

    newState->setTrainingFiles(training);
    newState->setTestFiles(testing);
  }

  if (getStrings() && getStrings()->size()) {
    std::vector<std::string>& dataset = *getStrings();
    boost::shared_ptr<std::vector<std::string> > training(new std::vector<std::string>());
    boost::shared_ptr<std::vector<std::string> > testing(new std::vector<std::string>());

    if (_FoldCount > 0) {

      if (dataset.size() % _FoldCount != 0) {
        dlog(Severity::Warning) << "The number of samples must be divisible by the FoldCount. Aborting!";
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

    newState->setTrainingStrings(training);
    newState->setTestStrings(testing);
  }
}

} /* namespace core */

} /* namespace gml */
