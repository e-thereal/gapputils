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

namespace convrbm {

BeginPropertyDefinitions(TensorReader)
  ReflectableBase(DefaultWorkflowElement<TensorReader>)

  WorkflowProperty(Filename, Input("File"), Filename(), FileExists())
  WorkflowProperty(MaxCount)
  WorkflowProperty(Tensors, Output("Ts"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())
  WorkflowProperty(TensorCount, NoParameter())

EndPropertyDefinitions

TensorReader::TensorReader() : _MaxCount(-1), _Width(0), _Height(0), _FilterCount(0), _TensorCount(0) {
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

  unsigned count;
  file.read((char*)&count, sizeof(count));

  if (getMaxCount() > 0)
    count = std::min((int)count, getMaxCount());

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > tensors(
      new std::vector<boost::shared_ptr<tensor_t> >());

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
    newState->setFilterCount(tensors->at(0)->size()[2]);
    newState->setTensorCount(tensors->size());
  }
}

} /* namespace convrbm */

} /* namespace gml */
