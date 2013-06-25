/*
 * DiagonalMatrix.cpp
 *
 *  Created on: 2013-05-24
 *      Author: tombr
 */

#include "DiagonalMatrix.h"

namespace gml {

namespace core {

BeginPropertyDefinitions(DiagonalMatrix)

  ReflectableBase(DefaultWorkflowElement<DiagonalMatrix>)

  WorkflowProperty(RowCount)
  WorkflowProperty(ColumnCount)
  WorkflowProperty(Matrix, Output())

EndPropertyDefinitions

DiagonalMatrix::DiagonalMatrix() : _RowCount(1), _ColumnCount(1) {
  setLabel("Diag");
}

void DiagonalMatrix::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  const int rows = getRowCount(), cols = getColumnCount();

  if (rows <= 0 || cols <= 0) {
    dlog(Severity::Warning) << "The number of rows and columns must be positive. Aborting!";
    return;
  }

  boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > matrix = boost::make_shared<std::vector<boost::shared_ptr<std::vector<double> > > >();
  for (int i = 0; i < rows; ++i) {
    boost::shared_ptr<std::vector<double> > row = boost::make_shared<std::vector<double> >(cols, 0);
    row->at(i) = 1.0;
    matrix->push_back(row);
  }

  newState->setMatrix(matrix);
}

} /* namespace core */

} /* namespace gml */
