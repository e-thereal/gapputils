/*
 * Identity.h
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_DIAGONALMATRIX_H_
#define GAPPUTILS_ML_DIAGONALMATRIX_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class DiagonalMatrix : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(DiagonalMatrix)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Matrix, boost::shared_ptr<std::vector<float> >)

private:
  mutable DiagonalMatrix* data;

public:
  DiagonalMatrix();
  virtual ~DiagonalMatrix();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_ML_DIAGONALMATRIX_H_ */
