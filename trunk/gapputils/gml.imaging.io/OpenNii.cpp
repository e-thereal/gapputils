/*
 * OpenNii.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#include "OpenNii.h"

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

BeginPropertyDefinitions(OpenNii)

  ReflectableBase(DefaultWorkflowElement<OpenNii>)

  WorkflowProperty(Filename, Input("Nii"), Filename("Nifti (*.nii *.nii.gz)"), FileExists())
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Image, Output("Img"))
  WorkflowProperty(Header, Output("Hdr"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(VoxelWidth, NoParameter(), Description("Voxel width in mm."))
  WorkflowProperty(VoxelHeight, NoParameter(), Description("Voxel height in mm."))
  WorkflowProperty(VoxelDepth, NoParameter(), Description("Voxel depth in mm."))

EndPropertyDefinitions

OpenNii::OpenNii() : _MaximumIntensity(1), _Width(0), _Height(0), _Depth(0), _VoxelWidth(0), _VoxelHeight(0), _VoxelDepth(0) {
  setLabel("Nii");
}

void OpenNii::update(IProgressMonitor* /*monitor*/) const {
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

  boost::shared_ptr<image_t> image(new image_t(hdr.dim[1], hdr.dim[2], hdr.dim[3],
      hdr.pixdim[1] * 1000,hdr.pixdim[2] * 1000, hdr.pixdim[3] * 1000));

  boost::shared_ptr<data_t> header(new data_t((long)hdr.vox_offset));
  std::copy((char*)&hdr, ((char*)&hdr) + sizeof(hdr), header->begin());
  if(!file.read((char*)&header->at(sizeof(hdr)), header->size() - sizeof(hdr))) {
    dlog(Severity::Warning) << "Error doing seekg() to " << (long)hdr.vox_offset << " in data file. Aborting!";
    return;
  }

  if (!file.read((char*)image->getData(), sizeof(float) * image->getCount())) {
    dlog(Severity::Warning) << "Error reading volume from NII file: " << file.gcount();
    return;
  }

  newState->setImage(image);
  newState->setHeader(header);

  newState->setWidth(hdr.dim[1]);
  newState->setHeight(hdr.dim[2]);
  newState->setDepth(hdr.dim[3]);

  newState->setVoxelWidth(hdr.pixdim[1]);
  newState->setVoxelHeight(hdr.pixdim[2]);
  newState->setVoxelDepth(hdr.pixdim[3]);
}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
