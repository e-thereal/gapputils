/*
 * AamReader.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamReader.h"

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

#include <cassert>
#include <cstdio>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Filename(), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ActiveAppearanceModel, Output("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamReader::AamReader() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamReader");

  Changed.connect(capputils::EventHandler<AamReader>(this, &AamReader::changedHandler));
}

AamReader::~AamReader() {
  if (data)
    delete data;
}

void AamReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<ActiveAppearanceModel> model(new ActiveAppearanceModel());

  int rowCount, columnCount, width, height, spCount, tpCount, apCount;
  int meanGridSize, meanImageSize, pgSize, piSize, ppSize;

  FILE* infile = fopen(getFilename().c_str(), "rb");
  if (!infile)
    return;

  assert(fread(&rowCount, sizeof(int), 1, infile) == 1);
  assert(fread(&columnCount, sizeof(int), 1, infile) == 1);
  assert(fread(&width, sizeof(int), 1, infile) == 1);
  assert(fread(&height, sizeof(int), 1, infile) == 1);
  assert(fread(&spCount, sizeof(int), 1, infile) == 1);
  assert(fread(&tpCount, sizeof(int), 1, infile) == 1);
  assert(fread(&apCount, sizeof(int), 1, infile) == 1);

  model->setRowCount(rowCount);
  model->setColumnCount(columnCount);
  model->setWidth(width);
  model->setHeight(height);
  model->setShapeParameterCount(spCount);
  model->setTextureParameterCount(tpCount);
  model->setAppearanceParameterCount(apCount);

  assert(fread(&meanGridSize, sizeof(int), 1, infile) == 1);
  boost::shared_ptr<std::vector<float> > meanGrid(new std::vector<float>(meanGridSize));
  assert((int)fread(&(*meanGrid.get())[0], sizeof(float), meanGridSize, infile) == meanGridSize);
  model->setMeanShape(meanGrid);

  assert(fread(&meanImageSize, sizeof(int), 1, infile) == 1);
  boost::shared_ptr<std::vector<float> > meanImage(new std::vector<float>(meanImageSize));
  assert((int)fread(&(*meanImage.get())[0], sizeof(float), meanImageSize, infile) == meanImageSize);
  model->setMeanTexture(meanImage);

  assert(fread(&pgSize, sizeof(int), 1, infile) == 1);
  boost::shared_ptr<std::vector<float> > pg(new std::vector<float>(pgSize));
  assert((int)fread(&(*pg.get())[0], sizeof(float), pgSize, infile) == pgSize);
  model->setShapeMatrix(pg);

  assert(fread(&piSize, sizeof(int), 1, infile) == 1);
  boost::shared_ptr<std::vector<float> > pi(new std::vector<float>(piSize));
  assert((int)fread(&(*pi.get())[0], sizeof(float), piSize, infile) == piSize);
  model->setTextureMatrix(pi);

  assert(fread(&ppSize, sizeof(int), 1, infile) == 1);
  boost::shared_ptr<std::vector<float> > pp(new std::vector<float>(ppSize));
  assert((int)fread(&(*pp.get())[0], sizeof(float), ppSize, infile) == ppSize);
  model->setAppearanceMatrix(pp);

  fclose(infile);

  data->setActiveAppearanceModel(model);
}

void AamReader::writeResults() {
  if (!data)
    return;

  setActiveAppearanceModel(data->getActiveAppearanceModel());
}

}

}
