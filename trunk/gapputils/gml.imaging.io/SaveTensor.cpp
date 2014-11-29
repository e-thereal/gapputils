/*
 * SaveTensor.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "SaveTensor.h"
#include <tbblas/serialize.hpp>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(SaveTensor)
  ReflectableBase(DefaultWorkflowElement<SaveTensor>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

SaveTensor::SaveTensor() {
  setLabel("Writer");
}

void SaveTensor::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  // Make sure that the path exists (optional)
  fs::path path(getFilename());
  fs::create_directories(path.parent_path());

  // Create a compressed file stream
  bio::filtering_ostream file;
  file.push(boost::iostreams::gzip_compressor());       // adds compression
  file.push(bio::file_descriptor_sink(getFilename()));  // specify the filename
  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for writing. Aborting!";
    return;
  }

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getTensors();

  // Write the number of tensors first (has to be one in your case)
  unsigned count = tensors.size();
  file.write((char*)&count, sizeof(count));


  for (size_t i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    // Only need a single call to serialize
    // matrix_t A;
    // tbblas::serialize(A, file);

    tbblas::serialize(*tensors[i], file);
    if (monitor)
      monitor->reportProgress(100.0 * i / count);
  }

  newState->setOutputName(getFilename());
}

} /* io */

} /* namespace imaging */

} /* namespace gml */
