/*
 * OpenNiiTensor.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: tombr
 */

#include "OpenNiiTensor.h"

#include <cstdio>
#include <fstream>
#include <algorithm>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include "nifti1.h"

namespace bio = boost::iostreams;

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(OpenNiiTensor)

  ReflectableBase(DefaultWorkflowElement<OpenNiiTensor>)

  WorkflowProperty(Filename, Input("Nii"), Filename("Nifti (*.nii *.nii.gz)"), FileExists())
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Tensor, Output("Ten"))
  WorkflowProperty(Header, Output("Hdr"))
  WorkflowProperty(Size, NoParameter())
  WorkflowProperty(VoxelSize, NoParameter(), Description("Voxel size in mm"))

EndPropertyDefinitions

OpenNiiTensor::OpenNiiTensor() : _MaximumIntensity(1) {
  setLabel("NiiTensor");
}

void OpenNiiTensor::update(IProgressMonitor* /*monitor*/) const {
  Logbook& dlog = getLogbook();

  nifti_1_header hdr;
  bio::filtering_istream file;

  if (getFilename().size() > 3 && getFilename().substr(getFilename().size() - 3, 3) == ".gz") {
    file.push(boost::iostreams::gzip_decompressor());
  }
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Error opening NII file. Aborting!";
    return;
  }

  if (!file.read((char*)&hdr, sizeof(hdr))) {
    dlog(Severity::Warning) << "Error reading NII header. Aborting!";
    return;
  }

  if (hdr.datatype != DT_FLOAT) {
    dlog(Severity::Warning) << "Only data type float is supported. Aborting!";
    return;
  }

  boost::shared_ptr<tensor_t> tensor(new tensor_t(hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]));

  boost::shared_ptr<data_t> header(new data_t((long)hdr.vox_offset));
  std::copy((char*)&hdr, ((char*)&hdr) + sizeof(hdr), header->begin());
  if(!file.read((char*)&header->at(sizeof(hdr)), header->size() - sizeof(hdr))) {
    dlog(Severity::Warning) << "Error doing seekg() to " << (long)hdr.vox_offset << " in data file. Aborting!";
    return;
  }

  if (!file.read((char*)tensor->data().data(), sizeof(float) * tensor->count())) {
    dlog(Severity::Warning) << "Error reading volume from NII file: " << file.gcount();
    return;
  }

  newState->setTensor(tensor);
  newState->setHeader(header);
  newState->setSize(tensor->size());
  newState->setVoxelSize(tbblas::seq(hdr.pixdim[1],hdr.pixdim[2], hdr.pixdim[3], hdr.pixdim[4]));
}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
