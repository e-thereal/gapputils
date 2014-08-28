/*
 * SaveNii.cpp
 *
 *  Created on: Aug 27, 2014
 *      Author: tombr
 */

#include "SaveNii.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include "nifti1.h"

namespace bio = boost::iostreams;

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(SaveNii)

  ReflectableBase(DefaultWorkflowElement<SaveNii>)

  WorkflowProperty(Image, Input("I"), NotNull<Type>())
  WorkflowProperty(Header, Input("H"))
  WorkflowProperty(Filename, Filename("Nifti (*.nii *.nii.gz"), NotEmpty<Type>())
  WorkflowProperty(OutpuName, Output("Name"))

EndPropertyDefinitions

SaveNii::SaveNii() {
  setLabel("Save");
}

void SaveNii::update(IProgressMonitor* /*monitor*/) const {
  Logbook& dlog = getLogbook();

  bio::filtering_ostream file;
  image_t& image = *getImage();

  if (getFilename().size() > 3 && getFilename().substr(getFilename().size() - 3, 3) == ".gz") {
    file.push(boost::iostreams::gzip_compressor());
  }
  file.push(bio::file_descriptor_sink(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Error opening NII file. Aborting!";
    return;
  }

  if (getHeader()) {
    data_t& header = *getHeader();
    if (!file.write(&header[0], header.size())) {
      dlog(Severity::Warning) << "Error writing header to Nii. Aborting!";
      return;
    }
  } else {
    data_t header(352);
    std::fill(header.begin(), header.end(), 0);

    nifti_1_header& hdr = *((nifti_1_header*)&header[0]);
    hdr.datatype = DT_FLOAT;

    for (size_t i = 0; i < 8; ++i) {
      hdr.dim[i] = 1;
      hdr.pixdim[i] = 1;
    }

    int dims = 0;

    for (size_t i = 0; i < 3; ++i) {
      dims += image.getSize()[i] > 1;
      hdr.dim[i + 1] = image.getSize()[i];
      hdr.pixdim[i + 1] = (float)image.getPixelSize()[i] / 1000.f;
    }
    hdr.dim[0] = dims;

    hdr.sizeof_hdr = 348;
    hdr.vox_offset = 352;
    hdr.scl_slope = 1;
    hdr.scl_inter = 0;
    hdr.magic[0] = 'n';
    hdr.magic[1] = '+';
    hdr.magic[2] = '1';
    hdr.bitpix = 32;
    hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_SEC;
    hdr.cal_min = *std::min_element(image.begin(), image.end());
    hdr.cal_max = *std::max_element(image.begin(), image.end());

    if (!file.write(&header[0], header.size())) {
      dlog(Severity::Warning) << "Error writing header to NII. Aborting!";
      return;
    }
  }

  if (!file.write((char*)image.getData(), sizeof(float) * image.getCount())) {
    dlog(Severity::Warning) << "Error writing volume to NII file.";
    return;
  }

  newState->setOutpuName(getFilename());
}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
