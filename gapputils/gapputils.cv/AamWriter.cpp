/*
 * AamWriter.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamWriter.h"

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

#include "AamReader.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamWriter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Output("Name"), Filename(), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

AamWriter::AamWriter() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamWriter");

  Changed.connect(capputils::EventHandler<AamWriter>(this, &AamWriter::changedHandler));
}

AamWriter::~AamWriter() {
  if (data)
    delete data;
}

void AamWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamWriter();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel())
    return;

  ActiveAppearanceModel* model = getActiveAppearanceModel().get();

  if (!model->getMeanShape() || !model->getMeanTexture() || !model->getShapeMatrix() ||
      !model->getTextureMatrix() || !model->getAppearanceMatrix() ||
      !model->getSingularTextureParameters() || !model->getSingularAppearanceParameters() ||
      !model->getSingularShapeParameters())
  {
    return;
  }

  FILE* outfile = fopen(getFilename().c_str(), "wb");
  int rowCount = model->getRowCount();
  int columnCount = model->getColumnCount();
  int width = model->getWidth();
  int height = model->getHeight();
  int spCount = model->getShapeParameterCount();
  int tpCount = model->getTextureParameterCount();
  int apCount = model->getAppearanceParameterCount();

  int meanGridSize = model->getMeanShape()->size();
  int meanImageSize = model->getMeanTexture()->size();
  int pgSize = model->getShapeMatrix()->size();
  int piSize = model->getTextureMatrix()->size();
  int ppSize = model->getAppearanceMatrix()->size();

  assert((int)model->getSingularShapeParameters()->size() == spCount);
  assert((int)model->getSingularTextureParameters()->size() == tpCount);
  assert((int)model->getSingularAppearanceParameters()->size() == apCount);

  if (!outfile)
    return;

  fwrite(&rowCount, sizeof(int), 1, outfile);
  fwrite(&columnCount, sizeof(int), 1, outfile);
  fwrite(&width, sizeof(int), 1, outfile);
  fwrite(&height, sizeof(int), 1, outfile);
  fwrite(&spCount, sizeof(int), 1, outfile);
  fwrite(&tpCount, sizeof(int), 1, outfile);
  fwrite(&apCount, sizeof(int), 1, outfile);

  fwrite(&meanGridSize, sizeof(int), 1, outfile);
  fwrite(&(*model->getMeanShape().get())[0], sizeof(float), meanGridSize, outfile);

  fwrite(&meanImageSize, sizeof(int), 1, outfile);
  fwrite(&(*model->getMeanTexture().get())[0], sizeof(float), meanImageSize, outfile);

  fwrite(&pgSize, sizeof(int), 1, outfile);
  fwrite(&(*model->getShapeMatrix().get())[0], sizeof(float), pgSize, outfile);

  fwrite(&piSize, sizeof(int), 1, outfile);
  fwrite(&(*model->getTextureMatrix().get())[0], sizeof(float), piSize, outfile);

  fwrite(&ppSize, sizeof(int), 1, outfile);
  fwrite(&(*model->getAppearanceMatrix().get())[0], sizeof(float), ppSize, outfile);

  fwrite(&(*model->getSingularShapeParameters())[0], sizeof(float), spCount, outfile);
  fwrite(&(*model->getSingularTextureParameters())[0], sizeof(float), tpCount, outfile);
  fwrite(&(*model->getSingularAppearanceParameters())[0], sizeof(float), apCount, outfile);

  fclose(outfile);

//  AamReader reader;
//  reader.setFilename(getFilename());
//  reader.execute(0);
//  reader.writeResults();

//  ActiveAppearanceModel* model2 = reader.getActiveAppearanceModel().get();
//  assert(model->getRowCount() == model2->getRowCount());
//  assert(model->getColumnCount() == model2->getColumnCount());
//  assert(model->getWidth() == model2->getWidth());
//  assert(model->getHeight() == model2->getHeight());
//  assert(model->getShapeParameterCount() == model2->getShapeParameterCount());
//  assert(model->getTextureParameterCount() == model2->getTextureParameterCount());
//  assert(model->getAppearanceParameterCount() == model2->getAppearanceParameterCount());
//
//  assert(model->getMeanGrid()->size() == model2->getMeanGrid()->size());
//  assert(model->getMeanImage()->size() == model2->getMeanImage()->size());
//  assert(model->getPrincipalGrids()->size() == model2->getPrincipalGrids()->size());
//  assert(model->getPrincipalImages()->size() == model2->getPrincipalImages()->size());
//  assert(model->getPrincipalParameters()->size() == model2->getPrincipalParameters()->size());
//
//  for (unsigned i = 0; i < model->getMeanGrid()->size(); ++i)
//    assert(model->getMeanGrid()->at(i) == model2->getMeanGrid()->at(i));
//  for (unsigned i = 0; i < model->getMeanImage()->size(); ++i)
//    assert(model->getMeanImage()->at(i) == model2->getMeanImage()->at(i));
//  for (unsigned i = 0; i < model->getPrincipalGrids()->size(); ++i)
//    assert(model->getPrincipalGrids()->at(i) == model2->getPrincipalGrids()->at(i));
//  for (unsigned i = 0; i < model->getPrincipalImages()->size(); ++i)
//    assert(model->getPrincipalImages()->at(i) == model2->getPrincipalImages()->at(i));
//  for (unsigned i = 0; i < model->getPrincipalParameters()->size(); ++i)
//    assert(model->getPrincipalParameters()->at(i) == model2->getPrincipalParameters()->at(i));
}

void AamWriter::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
