/*
 * Identity.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "DiagonalMatrix.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(DiagonalMatrix)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(Matrix, Output("Data"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

DiagonalMatrix::DiagonalMatrix() : _RowCount(1), _ColumnCount(1), data(0) {
  WfeUpdateTimestamp
  setLabel("DiagonalMatrix");

  Changed.connect(capputils::EventHandler<DiagonalMatrix>(this, &DiagonalMatrix::changedHandler));
}

DiagonalMatrix::~DiagonalMatrix() {
  if (data)
    delete data;
}

void DiagonalMatrix::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void DiagonalMatrix::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new DiagonalMatrix();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<std::vector<float> > matrix(new std::vector<float>(getRowCount() * getColumnCount(), 0.f));
  for (size_t i = 0; i < matrix->size(); i += getColumnCount() + 1)
    matrix->at(i) = 1.f;
  data->setMatrix(matrix);
}

void DiagonalMatrix::writeResults() {
  if (!data)
    return;

  setMatrix(data->getMatrix());
}

}

}
