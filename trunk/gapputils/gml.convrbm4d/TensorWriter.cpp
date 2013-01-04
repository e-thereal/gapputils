/*
 * TensorWriter.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "TensorWriter.h"
#include <tbblas/serialize.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(TensorWriter)
  ReflectableBase(DefaultWorkflowElement<TensorWriter>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

TensorWriter::TensorWriter() {
  setLabel("Writer");
}

void TensorWriter::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  bio::filtering_ostream file;
  file.push(boost::iostreams::gzip_compressor());
  file.push(bio::file_descriptor_sink(getFilename()));
  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for writing. Aborting!";
    return;
  }

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getTensors();

  unsigned count = tensors.size();
  file.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    tbblas::serialize(*tensors[i], file);
    if (monitor)
      monitor->reportProgress(100.0 * i / count);
  }

  newState->setOutputName(getFilename());
}

} /* namespace convrbm4d */

} /* namespace gml */
