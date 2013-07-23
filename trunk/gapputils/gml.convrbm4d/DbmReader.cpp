/*
 * DbmReader.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "DbmReader.h"

#include <capputils/Serializer.h>
#include <capputils/DeprecatedAttribute.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(DbmReader, Deprecated("Use gml.dbm.ModelReader instead."))

  ReflectableBase(DefaultWorkflowElement<DbmReader>)

  WorkflowProperty(Filename, Input("File"), Filename("Compressed DBM (*.dbm.gz)"), FileExists())
  WorkflowProperty(Model, Output("DBM"))
  WorkflowProperty(FilterWidth, NoParameter())
  WorkflowProperty(FilterHeight, NoParameter())
  WorkflowProperty(FilterDepth, NoParameter())
  WorkflowProperty(ChannelCount, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())

EndPropertyDefinitions

DbmReader::DbmReader() {
  setLabel("Reader");
}

void DbmReader::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  bio::filtering_istream file;
  file.push(boost::iostreams::gzip_decompressor());
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for reading. Aborting!";
    return;
  }

  boost::shared_ptr<DbmModel> model(new DbmModel());
  Serializer::ReadFromFile(*model, file);

  newState->setModel(model);

  if (model->getWeights()) {
    std::vector<int> widths, heights, depths, channels, filters;
    for (size_t iLayer = 0; iLayer < model->getWeights()->size(); ++iLayer) {

      if (model->getWeights()->at(iLayer) && model->getWeights()->at(iLayer)->size()) {
        widths.push_back(model->getWeights()->at(iLayer)->at(0)->size()[0]);
        heights.push_back(model->getWeights()->at(iLayer)->at(0)->size()[1]);
        depths.push_back(model->getWeights()->at(iLayer)->at(0)->size()[2]);
        channels.push_back(model->getWeights()->at(iLayer)->at(0)->size()[3]);
        filters.push_back(model->getWeights()->at(iLayer)->size());
      }
    }

    newState->setFilterWidth(widths);
    newState->setFilterHeight(heights);
    newState->setFilterDepth(depths);
    newState->setChannelCount(channels);
    newState->setFilterCount(filters);
  }
}

} /* namespace convrbm4d */

} /* namespace gml */
