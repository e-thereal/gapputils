/*
 * FindThreshold.cpp
 *
 *  Created on: Jan 07, 2015
 *      Author: tombr
 */

#include "FindThreshold.h"

namespace gml {

namespace encoder {

BeginPropertyDefinitions(FindThreshold)

  ReflectableBase(DefaultWorkflowElement<FindThreshold>)

  WorkflowProperty(InitialModel, Input("ENN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Metric, Enumerator<Type>())
  WorkflowProperty(VoxelSize)
  WorkflowProperty(TestThreshold)
//  WorkflowProperty(Model, Output("ENN"))
  WorkflowProperty(LesionLoadsGlobal, Output("LLG"))
  WorkflowProperty(LesionLoadsTest, Output("LLT"))
  WorkflowProperty(LesionLoadsOptimal, Output("LLO"))
  WorkflowProperty(LesionLoadsPredicted, Output("LLP"))
  WorkflowProperty(PPV, Output("PPV"))
  WorkflowProperty(TPR, Output("TPR"))

EndPropertyDefinitions

FindThreshold::FindThreshold() : _VoxelSize(tbblas::seq(1,1,1,1)), _TestThreshold(0.5) {
  setLabel("Thresh");
}

FindThresholdChecker findThresholdChecker;

} /* namespace nn */

} /* namespace gml */
