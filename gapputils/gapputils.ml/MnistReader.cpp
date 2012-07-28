/*
 * MnistReader.cpp
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#include "MnistReader.h"

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

//#include <thrust/reduce.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <boost/lambda/lambda.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(MnistReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Input("Name"), FileExists(), Filename(), Observe(Id), TimeStamp(Id))
  DefineProperty(MaxImageCount, Observe(Id), TimeStamp(Id))
  DefineProperty(MakeBinary, Observe(Id), TimeStamp(Id))
  DefineProperty(ImageCount, Observe(Id), TimeStamp(Id))
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(FeatureCount, Observe(Id), TimeStamp(Id))
  DefineProperty(Data, Output(), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

MnistReader::MnistReader()
 : _MaxImageCount(-1), _ImageCount(0), _RowCount(0), _ColumnCount(0), _FeatureCount(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("MnistReader");

  Changed.connect(capputils::EventHandler<MnistReader>(this, &MnistReader::changedHandler));
}

MnistReader::~MnistReader() {
  if (data)
    delete data;
}

void MnistReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

inline void swapEndian(unsigned int &val) {
  val = (val<<24) | ((val<<8) & 0x00ff0000) | ((val>>8) & 0x0000ff00) | (val>>24);
}


void MnistReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace boost::lambda;

  if (!data)
    data = new MnistReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  FILE* file = fopen(getFilename().c_str(), "rb");
  if (!file)
    return;

  unsigned int magic, imageCount, rowCount, columnCount;
  assert(fread(&magic, sizeof(int), 1, file) == 1);
  assert(fread(&imageCount, sizeof(int), 1, file) == 1);
  assert(fread(&rowCount, sizeof(int), 1, file) == 1);
  assert(fread(&columnCount, sizeof(int), 1, file) == 1);

  swapEndian(imageCount);
  swapEndian(rowCount);
  swapEndian(columnCount);

  if (getMaxImageCount() >= 0)
    imageCount = std::min(imageCount, (unsigned)getMaxImageCount());

  const unsigned count = imageCount * rowCount * columnCount;
  std::vector<unsigned char> bytes(count);
  boost::shared_ptr<std::vector<float> > floats(new std::vector<float>(count));

  assert(fread(&bytes[0], sizeof(unsigned char), count, file) == count);
  std::copy(bytes.begin(), bytes.end(), floats->begin());

  if (getMakeBinary()) {
    float mean = 0.f;
    for_each(floats->begin(), floats->end(), mean += _1);
    mean /= floats->size();
    for (int i = 0; i < floats->size(); ++i)
      floats->at(i) = floats->at(i) > mean;
  }

  fclose(file);

  data->setImageCount(imageCount);
  data->setRowCount(rowCount);
  data->setColumnCount(columnCount);
  data->setFeatureCount(rowCount * columnCount);
  data->setData(floats);
}

void MnistReader::writeResults() {
  if (!data)
    return;

  setImageCount(data->getImageCount());
  setRowCount(data->getRowCount());
  setColumnCount(data->getColumnCount());
  setFeatureCount(data->getFeatureCount());
  setData(data->getData());
}

}

}
