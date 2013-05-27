/*
 * DiagonalMatrix.h
 *
 *  Created on: 2013-05-24
 *      Author: tombr
 */

#ifndef GML_DIAGONALMATRIX_H_
#define GML_DIAGONALMATRIX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class DiagonalMatrix : public DefaultWorkflowElement<DiagonalMatrix> {

  InitReflectableClass(DiagonalMatrix)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Matrix, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)

public:
  DiagonalMatrix();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_DIAGONALMATRIX_H_ */
