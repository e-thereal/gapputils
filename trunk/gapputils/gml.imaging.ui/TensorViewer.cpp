/*
 * TensorViewer.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "TensorViewer.h"

#include <capputils/EventHandler.h>

namespace gml {

namespace imaging {

namespace ui {

int TensorViewer::tensorId;
int TensorViewer::tensorsId;
int TensorViewer::backgroundId;
int TensorViewer::currentTensorId;
int TensorViewer::currentSliceId;
int TensorViewer::minimumLengthId;
int TensorViewer::maximumLengthId;
int TensorViewer::resolutionId;

BeginPropertyDefinitions(TensorViewer)

  ReflectableBase(DefaultWorkflowElement<TensorViewer>)

  WorkflowProperty(Tensor, Input("T"), Dummy(tensorId = Id))
  WorkflowProperty(Tensors, Input("Ts"), Dummy(tensorsId = Id))
  WorkflowProperty(Background, Input("Bg"), Dummy(backgroundId = Id))
  WorkflowProperty(CurrentTensor, Dummy(currentTensorId = Id))
  WorkflowProperty(CurrentSlice, Dummy(currentSliceId = Id))
  WorkflowProperty(MinimumLength, Dummy(minimumLengthId = Id))
  WorkflowProperty(MaximumLength, Dummy(maximumLengthId = Id))
  WorkflowProperty(VisibleLength)
  WorkflowProperty(GridResolution, Dummy(resolutionId = Id))

EndPropertyDefinitions

TensorViewer::TensorViewer()
 : _CurrentTensor(0), _CurrentSlice(0), _MinimumLength(0.0), _MaximumLength(1.0), _VisibleLength(1.0)
{
  setLabel("Viewer");
}

TensorViewer::~TensorViewer() {
  if (dialog) {
    dialog->close();
  }
}

void TensorViewer::show() {
  if (!dialog) {
    dialog = boost::make_shared<TensorViewerDialog>(this);
  }
  dialog->show();
}

}

}

}
