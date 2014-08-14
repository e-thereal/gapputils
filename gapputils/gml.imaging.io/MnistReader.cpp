/*
 * MnistReader.cpp
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#include "MnistReader.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <boost/lambda/lambda.hpp>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(MnistReader)

  ReflectableBase(DefaultWorkflowElement<MnistReader>)

  WorkflowProperty(ImageFile, Input("I"), FileExists(), Filename())
  WorkflowProperty(LabelFile, Input("L"), Filename(), FileExists())
  WorkflowProperty(MaxImageCount)
  WorkflowProperty(SelectedDigits)
  WorkflowProperty(MakeBinary, Flag())
  WorkflowProperty(Images, Output("I"))
  WorkflowProperty(Labels, Output("L"))
  WorkflowProperty(ImageCount, NoParameter())
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())

EndPropertyDefinitions

MnistReader::MnistReader() : _MaxImageCount(-1), _SelectedDigits(10), _ImageCount(0), _Width(0), _Height(0) {
  setLabel("Mnist");

  for (int i = 0; i < 10; ++i)
    _SelectedDigits[i] = i;
}

inline void swapEndian(unsigned int &val) {
  val = (val<<24) | ((val<<8) & 0x00ff0000) | ((val>>8) & 0x0000ff00) | (val>>24);
}

void MnistReader::update(IProgressMonitor* /*monitor*/) const {
  using namespace boost::lambda;

  Logbook& dlog = getLogbook();

  // try to read labels
  boost::shared_ptr<std::vector<unsigned char> > labels;
  std::vector<int> selectedDigits = getSelectedDigits();

  if (getLabelFile().size()) {
    FILE* file = fopen(getLabelFile().c_str(), "rb");

    unsigned int magic, imageCount;
    assert(fread(&magic, sizeof(int), 1, file) == 1);
    assert(magic == 0x01080000);
    assert(fread(&imageCount, sizeof(int), 1, file) == 1);

    swapEndian(imageCount);

    labels = boost::make_shared<std::vector<unsigned char> >(imageCount);
    assert(fread(&(*labels)[0], sizeof(unsigned char), imageCount, file) == imageCount);

    fclose(file);
  }

  FILE* file = fopen(getImageFile().c_str(), "rb");
  if (!file)
    return;

  unsigned int magic, imageCount, rowCount, columnCount;
  assert(fread(&magic, sizeof(int), 1, file) == 1);
  assert(magic == 0x03080000);
  assert(fread(&imageCount, sizeof(int), 1, file) == 1);
  assert(fread(&rowCount, sizeof(int), 1, file) == 1);
  assert(fread(&columnCount, sizeof(int), 1, file) == 1);

  swapEndian(imageCount);
  swapEndian(rowCount);
  swapEndian(columnCount);

  if (labels && labels->size() != imageCount) {
    dlog(Severity::Warning) << "Number of images does not match the number of labels. Aborting!";
    return;
  }

  size_t maxImageCount = (getMaxImageCount() > 0 ? getMaxImageCount() : imageCount);

  const size_t count = rowCount * columnCount;

  std::vector<unsigned char> bytes(count);
  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > images(
      new std::vector<boost::shared_ptr<image_t> >());

  boost::shared_ptr<std::vector<double> > usedLabels(new std::vector<double>());

  for (unsigned i = 0; i < imageCount; ++i) {

    if (images->size() >= maxImageCount)
      break;

    assert(fread(&bytes[0], sizeof(unsigned char), count, file) == count);

    if (labels && std::find(selectedDigits.begin(), selectedDigits.end(), (int)labels->at(i)) == selectedDigits.end()) {
      continue;
    }

    boost::shared_ptr<image_t> image(new image_t(columnCount, rowCount, 1));
    std::copy(bytes.begin(), bytes.end(), image->begin());

    if (getMakeBinary()) {
      float mean = 0.f;
      std::for_each(image->begin(), image->end(), mean += _1);
      mean /= image->getCount();
      for (size_t i = 0; i < image->getCount(); ++i)
        image->getData()[i] = image->getData()[i] > mean;
    }

    images->push_back(image);

    if (labels)
      usedLabels->push_back(labels->at(i));
  }

  fclose(file);

  newState->setImages(images);
  newState->setImageCount(images->size());
  newState->setWidth(columnCount);
  newState->setHeight(rowCount);

  if (labels) {
    newState->setLabels(usedLabels);
  }
}

}

}

}
