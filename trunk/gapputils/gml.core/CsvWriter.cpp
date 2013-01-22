#include "CsvWriter.h"

#include <fstream>

namespace gml {

namespace core {

BeginPropertyDefinitions(CsvWriter)

  ReflectableBase(workflow::DefaultWorkflowElement<CsvWriter>)

  WorkflowProperty(Data, Input("D"))
  WorkflowProperty(FlatData, Input("F"))
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

  if (getData()) {
    std::vector<boost::shared_ptr<std::vector<double> > >& data = *getData();
    for (size_t iRow = 0; iRow < data.size(); ++iRow) {
      std::vector<double>& row = *data[iRow];
      if (row.size())
        outfile << row[0];
      for (size_t iCol = 1; iCol < row.size(); ++iCol) {
        outfile << ", " << row[iCol];
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
        if ((i + 1) % columnCount == 0)
          outfile << rows[i] << "\n";
        else
          outfile << rows[i] << ", ";
      }
    }
  }

  outfile.close();
  newState->setOutputName(getFilename());
}

}

}
