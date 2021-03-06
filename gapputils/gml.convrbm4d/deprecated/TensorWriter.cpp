/*
 * TensorWriter.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "TensorWriter.h"
#include <tbblas/serialize.hpp>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(TensorWriter, Renamed("gml::imaging::io::SaveTensor"), Deprecated("Use gml::imaging::io::SaveTensor instead."))
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

  fs::path path(getFilename());
  fs::create_directories(path.parent_path());

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
