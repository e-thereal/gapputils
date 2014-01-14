#include "CsvWriter.h"

#include <fstream>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace gml {

namespace core {

BeginPropertyDefinitions(CsvWriter)

  ReflectableBase(workflow::DefaultWorkflowElement<CsvWriter>)

  WorkflowProperty(Data, Input("D"))
  WorkflowProperty(FlatData, Input("F"))
  WorkflowProperty(OnlyRowNames, Description("If checked, only the row names are written. (The values of Data and FlatData are ignored)"), Flag())
  WorkflowProperty(RowNames, Input("Ids"), Description("(Optional) Intended to be used as a row ID. Adds one column to the beginning of the output."))
  WorkflowProperty(Header, Description("If not empty, the header is added before the first row of the output."))
  WorkflowProperty(ColumnCount, Description("The number of columns is only required when flat data is used."))
  WorkflowProperty(Filename, Filename(), NotEqual<Type>(""))
  WorkflowProperty(OutputName, Output("Csv"))

EndPropertyDefinitions

CsvWriter::CsvWriter() : _ColumnCount(0) {
  setLabel("CsvWriter");
}

void CsvWriter::update(workflow::IProgressMonitor* monitor) const {
  std::ofstream outfile(getFilename().c_str());
  Logbook& dlog = getLogbook();

  fs::path path(getFilename());
  fs::create_directories(path.parent_path());

  size_t iRowName = 0;

  if (getHeader().size())
    outfile << getHeader() << std::endl;

  if (getOnlyRowNames()) {
    if (!getRowNames() || getRowNames()->size() == 0) {
      dlog(Severity::Warning) << "No row names given. Aborting!";
      return;
    }
    std::vector<std::string>& names = *getRowNames();
    for (size_t i = 0; i < names.size(); ++i) {
      outfile << names[i] << std::endl;
    }
    outfile.close();
    newState->setOutputName(getFilename());
    return;
  }

  if (getRowNames()) {
    size_t rowCount = 0;
    if (getData())
      rowCount += getData()->size();
    if (getFlatData() && getColumnCount() > 0 && getFlatData()->size() % getColumnCount() == 0)
      rowCount += getFlatData()->size() / getColumnCount();
    if (getRowNames()->size() != rowCount) {
      dlog(Severity::Warning) << "Number of row names (" << getRowNames()->size() << ") doens't match number of rows (" << rowCount << "). Aborting!";
      return;
    }
  }

  if (getData()) {
    std::vector<boost::shared_ptr<std::vector<double> > >& data = *getData();
    for (size_t iRow = 0; iRow < data.size(); ++iRow) {
      std::vector<double>& row = *data[iRow];
      if (getRowNames()) {
        outfile << getRowNames()->at(iRowName++);
        if (row.size())
          outfile << ",";
      }
      if (row.size())
        outfile << row[0];
      for (size_t iCol = 1; iCol < row.size(); ++iCol) {
        outfile << "," << row[iCol];
      }
      outfile << std::endl;
    }
  }

  if (getFlatData()) {
    if (getColumnCount() <= 0) {
      dlog(Severity::Warning) << "Column count must be greater than 0 in order to use flat data.";
    } else {
      const size_t columnCount = getColumnCount();
      std::vector<double>& rows = *getFlatData();
      for (size_t i = 0; i < rows.size(); ++i) {
        if (i % columnCount == 0)
          outfile << getRowNames()->at(iRowName++) << ",";
        if ((i + 1) % columnCount == 0)
          outfile << rows[i] << "\n";
        else
          outfile << rows[i] << ",";
      }
    }
  }

  outfile.close();
  newState->setOutputName(getFilename());
}

}

}
