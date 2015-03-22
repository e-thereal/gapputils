/*
 * TestThreshold2.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: tombr
 */

#include "TestThreshold2.h"

#include <tbblas/math.hpp>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(TestThreshold2)

  ReflectableBase(DefaultWorkflowElement<TestThreshold2>)

  WorkflowProperty(TrainingMaps, Input("TM"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TrainingLabels, Input("TL"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TestMaps, Input("EM"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TestLabels, Input("EL"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TrainTPR, Output("tTPR"))
  WorkflowProperty(TrainPPV, Output("tPPV"))
  WorkflowProperty(TrainDSC, Output("tDSC"))
  WorkflowProperty(TestTPR, Output("eTPR"))
  WorkflowProperty(TestPPV, Output("ePPV"))
  WorkflowProperty(TestDSC, Output("eDSC"))

EndPropertyDefinitions

TestThreshold2::TestThreshold2() {
  setLabel("Test");
}

void TestThreshold2::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef host_tensor_t::value_t value_t;

  v_host_tensor_t& maps = *getTrainingMaps();
  v_host_tensor_t& labels = *getTrainingLabels();

  v_host_tensor_t& testMaps = *getTestMaps();
  v_host_tensor_t& testLabels = *getTestLabels();

  boost::shared_ptr<data_t> tTPRs(new data_t(labels.size()));
  boost::shared_ptr<data_t> tPPVs(new data_t(labels.size()));
  boost::shared_ptr<data_t> tDSCs(new data_t(labels.size()));

  boost::shared_ptr<data_t> eTPRs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> ePPVs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> eDSCs(new data_t(testLabels.size()));

  if (maps.size() != labels.size()) {
    dlog(Severity::Warning) << "Need to have same number of input samples and labels. Aborting!";
    return;
  }

  if (maps[0]->size() != labels[0]->size()) {
    dlog(Severity::Warning) << "Maps and labels need to have the same dimension. Aborting!";
    return;
  }

  if (maps[0]->size() != testMaps[0]->size() || labels[0]->size() != testLabels[0]->size() || testMaps.size() != testLabels.size()) {
    dlog(Severity::Warning) << "Invalid test data. Aborting!";
    return;
  }

  value_t bestThreshold = 0, bestMetric = 0, currentMetric;
  for (value_t t = 0; t <= 1; t += 0.05) {

    value_t metric = 0;
    for (size_t iSample = 0; iSample < labels.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
      host_tensor_t& label = *labels[iSample];
      host_tensor_t& pred = *maps[iSample];

      metric += 2 * sum((label > 0.5) * (pred > t)) / (sum(label > 0.5) + sum(pred > t));
    }
    currentMetric = metric / labels.size();

    dlog(Severity::Trace) << "metric at " << t << " = " << currentMetric;

    if (currentMetric > bestMetric) {
      bestMetric = currentMetric;
      bestThreshold = t;
    }

    if (monitor)
      monitor->reportProgress(100. * (t + 0.05));
  }
  dlog(Severity::Message) << "Best threshold is " << bestThreshold << " (DSC = " << bestMetric << ")";

  value_t meanTPR, meanPPV, meanDSC;

  meanTPR = meanPPV = meanDSC = 0;

  // Calculate performance using a global threshold
  for (size_t iSample = 0; iSample < maps.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    host_tensor_t& pred = *maps[iSample];

    meanTPR += tTPRs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + 1e-8);
    meanPPV += tPPVs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(pred > bestThreshold);
    meanDSC += tDSCs->at(iSample) = 2 * sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + sum(pred > bestThreshold) + 1e-8);
  }

  dlog(Severity::Message) << "Training set: TPR = " << meanTPR / maps.size() << ", PPV = " << meanPPV / maps.size() << ", DSC = " << meanDSC / maps.size();

  meanTPR = meanPPV = meanDSC = 0;

  // Calculate performance using a global threshold
  for (size_t iSample = 0; iSample < testMaps.size(); ++iSample) {
    host_tensor_t& label = *testLabels[iSample];
    host_tensor_t& pred = *testMaps[iSample];

    meanTPR += eTPRs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + 1e-8);
    meanPPV += ePPVs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(pred > bestThreshold);
    meanDSC += eDSCs->at(iSample) = 2 * sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + sum(pred > bestThreshold) + 1e-8);
  }

  dlog(Severity::Message) << "Test set: TPR = " << meanTPR / testMaps.size() << ", PPV = " << meanPPV / testMaps.size() << ", DSC = " << meanDSC / testMaps.size();

  newState->setTrainTPR(tTPRs);
  newState->setTrainPPV(tPPVs);
  newState->setTrainDSC(tDSCs);

  newState->setTestTPR(eTPRs);
  newState->setTestPPV(ePPVs);
  newState->setTestDSC(eDSCs);
}

} /* namespace encoder */

} /* namespace gml */
