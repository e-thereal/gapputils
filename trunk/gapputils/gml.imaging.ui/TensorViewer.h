/*
 * TensorViewer.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GML_TENSORVIEWER_H_
#define GML_TENSORVIEWER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>
#include <tbblas/tensor.hpp>

#include "TensorViewerDialog.h"

namespace gml {

namespace imaging {

namespace ui {

class TensorViewerWidget;

class TensorViewer : public DefaultWorkflowElement<TensorViewer> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  friend class TensorViewerWidget;

  InitReflectableClass(TensorViewer)

  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Tensors, boost::shared_ptr<v_tensor_t>)
  Property(Background, boost::shared_ptr<image_t>)
  Property(CurrentTensor, int)
  Property(CurrentSlice, int)
  Property(MinimumLength, double)
  Property(MaximumLength, double)
  Property(VisibleLength, double)

public:
  static int tensorId, tensorsId, backgroundId, currentTensorId, currentSliceId, minimumLengthId, maximumLengthId;

private:
  boost::shared_ptr<TensorViewerDialog> dialog;

public:
  TensorViewer();
  virtual ~TensorViewer();

  virtual void show();
};

}

}

}

#endif /* GML_TENSORVIEWER_H_ */
