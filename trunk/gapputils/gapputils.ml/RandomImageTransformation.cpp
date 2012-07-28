/*
 * RandomImageTransformation.cpp
 *
 *  Created on: Nov 30, 2011
 *      Author: tombr
 */

#include "RandomImageTransformation.h"

#include <capputils/EnumeratorAttribute.h>
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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>

#include <culib/CudaImage.h>
#include <culib/transform.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RandomImageTransformation)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Input, Input("Img"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(Transformation, Enumerator<TransformationType>(), Observe(Id), TimeStamp(Id))
  DefineProperty(XRange, Observe(Id), TimeStamp(Id))
  DefineProperty(YRange, Observe(Id), TimeStamp(Id))
  DefineProperty(ZRange, Observe(Id), TimeStamp(Id))
  DefineProperty(Output, Output("Img"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

RandomImageTransformation::RandomImageTransformation() : _RowCount(1), _ColumnCount(1), data(0) {
  WfeUpdateTimestamp
  setLabel("RandomImageTransformation");

  _XRange.resize(2);
  _XRange[0] = -1;
  _XRange[1] = 1;
  _YRange.resize(2);
  _YRange[0] = -1;
  _YRange[1] = 1;

  Changed.connect(capputils::EventHandler<RandomImageTransformation>(this, &RandomImageTransformation::changedHandler));
}

RandomImageTransformation::~RandomImageTransformation() {
  if (data)
    delete data;
}

void RandomImageTransformation::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RandomImageTransformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RandomImageTransformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInput() || getRowCount() < 1 || getColumnCount() < 1 || getXRange().size() != 2 || getYRange().size() != 2)
    return;

  std::vector<double>& input = *getInput();

  const int rowCount = getRowCount();
  const int columnCount = getColumnCount();
  const int featureCount = rowCount * columnCount;
  const int count = (int)input.size();
  const int sampleCount = count / featureCount;
  const std::vector<int>& xrange = getXRange();
  const std::vector<int>& yrange = getYRange();
  const std::vector<int>& zrange = getZRange();

  if (count % featureCount != 0)
    return;

  boost::shared_ptr<std::vector<double> > output(new std::vector<double>(count));
  for (int sliceOffset = 0; sliceOffset < count; sliceOffset += featureCount) {
    // Get random transformation parameters
    // apply transformation

    float dx = (double)rand() / (double)RAND_MAX * (xrange[1] - xrange[0]) + xrange[0];
    float dy = (double)rand() / (double)RAND_MAX * (yrange[1] - yrange[0]) + yrange[0];
    float dz = (double)rand() / (double)RAND_MAX * (zrange[1] - zrange[0]) + zrange[0];

    fmatrix4 mat = make_fmatrix4_identity();

    switch(getTransformation()) {
    case TransformationType::Translation:
      mat = make_fmatrix4_translation(dx, dy, 0.f);
      break;

    case TransformationType::Rotation:
      mat = make_fmatrix4_translation(columnCount/2, rowCount/2,0) * make_fmatrix4_rotationZ(dz) *
            make_fmatrix4_translation(-columnCount/2, -rowCount/2,0);
      break;

    case TransformationType::Scaling:
      mat = make_fmatrix4_translation(columnCount/2, rowCount/2,0) * make_fmatrix4_scaling(dx, dy, 1.0f)
          * make_fmatrix4_translation(-columnCount/2, -rowCount/2,0);
      break;

    case TransformationType::Rigid:
      mat = make_fmatrix4_translation(dx + columnCount/2, dy + rowCount/2,0) * make_fmatrix4_rotationZ(dz) *
            make_fmatrix4_translation(-columnCount/2, -rowCount/2,0);
    }

    culib::CudaImage image(dim3(columnCount, rowCount));
    std::copy(input.begin() + sliceOffset, input.begin() + sliceOffset + featureCount, image.getWorkingCopy());

    culib::transform3D(image.getDevicePointer(), image.getCudaArray(), image.getSize(), mat);
    image.saveDeviceToWorkingCopy();
    std::copy(image.getWorkingCopy(), image.getWorkingCopy() + featureCount, output->begin() + sliceOffset);
  }

  data->setOutput(output);
}

void RandomImageTransformation::writeResults() {
  if (!data)
    return;

  setOutput(data->getOutput());
}

}

}
