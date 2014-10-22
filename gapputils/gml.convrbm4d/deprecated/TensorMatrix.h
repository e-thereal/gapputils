/*
 * TensorMatrix.h
 *
 *  Created on: Apr 26, 2013
 *      Author: tombr
 */

#ifndef GML_TENSORMATRIX_H_
#define GML_TENSORMATRIX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "../Model.h"

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(TilingPlane, Axial, Sagittal, Coronal);

class TensorMatrix : public DefaultWorkflowElement<TensorMatrix> {

  typedef model_t::host_tensor_t tensor_t;

  InitReflectableClass(TensorMatrix)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(MaxCount, int)
  Property(ColumnCount, int)
  Property(TilingPlane, TilingPlane)
  Property(TensorMatrix, boost::shared_ptr<tensor_t>)

public:
  TensorMatrix();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_TENSORMATRIX_H_ */
