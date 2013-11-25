/*
 * CopyNiftiHeader.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: tombr
 */

#include "CopyNiftiHeader.h"

#include <iostream>
#include <fstream>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(CopyNiftiHeader)

  ReflectableBase(DefaultWorkflowElement<CopyNiftiHeader>)

  WorkflowProperty(InputName, Input("In"), Filename(), FileExists())
  WorkflowProperty(HeaderName, Filename(), FileExists())
  WorkflowProperty(OutputName, Filename(), NotEmpty<Type>())
  WorkflowProperty(Output, Output("Out"), Filename())

EndPropertyDefinitions

CopyNiftiHeader::CopyNiftiHeader() {
  setLabel("CopyHeader");
}

void CopyNiftiHeader::update(IProgressMonitor* /*monitor*/) const {
  Logbook& dlog = getLogbook();

  std::ifstream input(getInputName().c_str(), std::ios::binary);
  std::ifstream header(getHeaderName().c_str(), std::ios::binary);

  const int hdrSize = 348;

  int size = 0;
  input.read((char*)&size, sizeof(size));
  if (size != hdrSize) {
    dlog(Severity::Warning) << "Invalid header size " << size << " of input file. Aborting!";
    return;
  }
  size = 0;
  header.read((char*)&size, sizeof(size));
  if (size != hdrSize) {
    dlog(Severity::Warning) << "Invalid header size " << size << " of header file. Aborting!";
    return;
  }

  input.seekg(0, std::ios::end);
  std::ifstream::pos_type imgSize = (int)input.tellg() - hdrSize;

  char* hdrBuffer = new char[hdrSize];
  char* imgBuffer = new char[imgSize];

  header.seekg(0);
  header.read(hdrBuffer, hdrSize);

  input.seekg(hdrSize);
  input.read(imgBuffer, imgSize);

  input.close();
  header.close();

  // Buffer need to be read before any writing takes place.
  std::ofstream output(getOutputName().c_str(), std::ios::binary);
  output.write(hdrBuffer, hdrSize);
  output.write(imgBuffer, imgSize);
  output.close();

  delete[] hdrBuffer;
  delete[] imgBuffer;

  newState->setOutput(getOutputName());
}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
