/*
 * TensorReader.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "TensorReader.h"

#include <tbblas/serialize.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(TensorReader)
  ReflectableBase(DefaultWorkflowElement<TensorReader>)

  WorkflowProperty(Filename, Input("File"), Filename(), FileExists())
  WorkflowProperty(FirstIndex)
  WorkflowProperty(MaxCount)
  WorkflowProperty(Tensors, Output("Ts"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())
  WorkflowProperty(TensorCount, NoParameter())

EndPropertyDefinitions

TensorReader::TensorReader() : _FirstIndex(0), _MaxCount(-1), _Width(0), _Height(0), _Depth(0), _FilterCount(0), _TensorCount(0) {
  setLabel("Reader");
}

void TensorReader::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  bio::filtering_istream file;
  file.push(boost::iostreams::gzip_decompressor());
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for reading. Aborting!";
    return;
  }

  int first = getFirstIndex();

  unsigned count;
  file.read((char*)&count, sizeof(count));

  if ((int)count <= first) {
    dlog(Severity::Warning) << "Invalid FirstIndex. Aborting!";
    return;
  }

  if (getMaxCount() > 0)
    count = std::min((int)count - first, getMaxCount());

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > tensors(
      new std::vector<boost::shared_ptr<tensor_t> >());

  {
    boost::shared_ptr<tensor_t> tensor(new tensor_t());
    for (int i = 0; i < first; ++i) {
      tbblas::deserialize(file, *tensor);
    }
  }

  for (unsigned i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    boost::shared_ptr<tensor_t> tensor(new tensor_t());
    tbblas::deserialize(file, *tensor);
    tensors->push_back(tensor);
    if (monitor)
      monitor->reportProgress(100.0 * i / count);
  }
  newState->setTensors(tensors);

  if (tensors->size()) {
    newState->setWidth(tensors->at(0)->size()[0]);
    newState->setHeight(tensors->at(0)->size()[1]);
    newState->setDepth(tensors->at(0)->size()[2]);
    newState->setFilterCount(tensors->at(0)->size()[3]);
    newState->setTensorCount(tensors->size());
  }
}

} /* namespace convrbm4d */

} /* namespace gml */
