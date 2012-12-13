/*
 * AverageMifs.cpp
 *
 *  Created on: Dec 11, 2012
 *      Author: tombr
 */

#include "AverageMifs.h"

#include <CMIF.hpp>
#include <CSlice.hpp>
#include <CProcessInfo.hpp>
#include <CChannel.hpp>

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AverageMifs)

  ReflectableBase(DefaultWorkflowElement<AverageMifs>)

  WorkflowProperty(MifNames, Input("Mifs"), Filename("*.MIF", true), Enumerable<Type, false>(), FileExists(), NotEmpty<Type>())
  WorkflowProperty(OutputName, NotEmpty<Type>())
  WorkflowProperty(Output, Output("Mif"))

EndPropertyDefinitions

AverageMifs::AverageMifs() {
  setLabel("Average");

  static char** argv = new char*[1];
  argv[0] = "AverageMifs";
  MSMRI::CProcessInfo::getInstance().getCommandLine(1, argv);
}

AverageMifs::~AverageMifs() { }

void AverageMifs::update(IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;

  std::vector<std::string> mifnames = getMifNames();

  CMIF output(mifnames[0]);
  const int width = output.getColumnCount();
  const int height = output.getRowCount();
  const int depth = output.getSliceCount();

  CMIF::pixelArray out = output.getRawData();
  for (int z = 1; z <= depth; ++z)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        out[z][y][x] = 0;

  for (size_t i = 0; i < mifnames.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    CMIF input(mifnames[i]);

    assert(width == input.getColumnCount());
    assert(height == input.getRowCount());
    assert(depth == input.getSliceCount());

    CMIF::pixelArray in = input.getRawData();
    for (int z = 1; z <= depth; ++z)
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
          out[z][y][x] += in[z][y][x];

    if (monitor)
      monitor->reportProgress(100. * i / mifnames.size());
  }

  for (int z = 1; z <= depth; ++z)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        out[z][y][x] /= mifnames.size();

  output.writeToFile(getOutputName(), true);
  newState->setOutput(getOutputName());
}

} /* namespace cv */
} /* namespace gapputils */
