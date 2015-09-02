#include "CsvWriter.h"

#include <capputils/attributes/MergeAttribute.h>

#include <fstream>

#include <boost/filesystem.hpp>
#include <cstdlib>

namespace fs = boost::filesystem;

namespace gml {

namespace core {

BeginPropertyDefinitions(CsvWriter)

  ReflectableBase(workflow::DefaultWorkflowElement<CsvWriter>)

  WorkflowProperty(Data, Input("D"), Merge<Type>())
  WorkflowProperty(FlatData, Input("F"), Merge<Type>())
  WorkflowProperty(OnlyRowNames, Description("If checked, only the row names are written. (The values of Data and FlatData are ignored)"), Flag())
  WorkflowProperty(RowNames, Input("Ids"), Description("(Optional) Intended to be used as a row ID. Adds one column to the beginning of the output."))
  WorkflowProperty(Header, Description("If not empty, the header is added before the first row of the output."))
  WorkflowProperty(ColumnCount, Description("The number of columns is only required when flat data is used."))
  WorkflowProperty(Filename, Filename(), NotEqual<Type>(""))
  WorkflowProperty(OutputName, Output("Csv"))

EndPropertyDefinitions

CsvWriter::CsvWriter() : _OnlyRowNames(false) {
  setLabel("CsvWriter");
}

int count_commas(const std::string str) {
  int count = 0;
  for (size_t i = 0; i < str.size(); ++i)
    if (str[i] == ',')
      ++count;
  return count;
}

void CsvWriter::update(workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  fs::path path(getFilename());
  if (!path.parent_path().empty() && !fs::exists(path.parent_path())) {
    fs::create_directories(path.parent_path());
  }

  std::ofstream outfile(getFilename().c_str());

  if (getData() && getFlatData()) {
    dlog(Severity::Warning) << "Only one of data or flat data may be given at a time. Aborting!";
    return;
  }

  if (getOnlyRowNames()) {
    if (getHeader().size())
      outfile << getHeader() << std::endl;

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

  // Get number of rows
  size_t rowCount = 0, columnCount = 0;
  if (getData() && getData()->size() && getData()->at(0)) {
    vv_data_t& data = *getData();

    rowCount = data[0]->size();
    for (size_t i = 1; i < data.size(); ++i) {
      if (!data[i] || data[i]->size() != rowCount) {
        dlog(Severity::Warning) << "Row counts don't match for all inputs. Aborting!";
        return;
      }
    }

    if (data[0]->size())
      columnCount = data[0]->at(0)->size();
  }

  if (getFlatData() && getFlatData()->size()) {
    // get number of flat rows
    if (_ColumnCount.size() != getFlatData()->size()) {
      dlog(Severity::Warning) << "Column counts must be specified for each flat data vector. Aborting!";
      return;
    }

    for (size_t i = 0; i < getFlatData()->size(); ++i) {
      if (!getFlatData()->at(i) || getFlatData()->at(i)->size() % _ColumnCount[i] != 0) {
        dlog(Severity::Warning) << "The size of the flat data vectors must be divisible by the column count of that vector. Aborting!";
        return;
      }
    }

    rowCount = getFlatData()->at(0)->size() / _ColumnCount[0];
    for (size_t i = 1; i < getFlatData()->size(); ++i) {
      if (rowCount != getFlatData()->at(i)->size() / _ColumnCount[i]) {
        dlog(Severity::Warning) << "Row counts of flat vectors don't match. Aborting!";
        return;
      }
    }

    columnCount = 0;
    for (size_t i = 0; i < _ColumnCount.size(); ++i)
      columnCount += _ColumnCount[i];
  }

  if (getHeader().size()) {
    std::string header = getHeader(), prefix;
    size_t pos = header.find_last_of(",");
    if (pos != std::string::npos) {
      prefix = header.substr(pos + 1);
      header = header.substr(0, pos);
    } else {
      prefix = header;
      header = "";
    }

    std::stringstream newheader;

    int commaCount = count_commas(header);
    if (header.size()) {
      newheader << header << ",";
      ++commaCount;
    }

    int totalCommaCount = columnCount;
    if (getRowNames() && getRowNames()->size() && getRowNames()->at(0).size()) {
      totalCommaCount += count_commas(getRowNames()->at(0)) + 1;
    }

    for (int i = 0; i < totalCommaCount - commaCount; ++i) {
      newheader << prefix;
      if (totalCommaCount - commaCount > 1)
        newheader << i + 1;
      if (i + 1 < totalCommaCount - commaCount)
        newheader << ",";
    }
    outfile << newheader.str() << std::endl;
  }

  if (getRowNames()) {
    if (getRowNames()->size() != rowCount) {
      dlog(Severity::Warning) << "Number of row names (" << getRowNames()->size() << ") doens't match number of rows (" << rowCount << "). Aborting!";
      return;
    }
  }

  if (getData() && getData()->size() && getData()->at(0)) {
    vv_data_t& data = *getData();

    for (size_t iRow = 0; iRow < rowCount; ++iRow) {
      if (getRowNames()) {
        outfile << getRowNames()->at(iRow) << ",";
      }
      for (size_t i = 0; i < data.size(); ++i) {
        std::vector<double>& row = *data[i]->at(iRow);
        for (size_t iCol = 0; iCol < row.size(); ++iCol) {
          if (iCol || i)
            outfile << ",";
          outfile << row[iCol];
        }
      }
      outfile << std::endl;
    }
  }

  if (getFlatData() && getFlatData()->size()) {
    v_data_t& data = *getFlatData();

    for (size_t iRow = 0; iRow < rowCount; ++iRow) {
      if (getRowNames())
        outfile << getRowNames()->at(iRow) << ",";
      for (size_t i = 0; i < data.size(); ++i) {
        for (int iCol = 0; iCol < _ColumnCount[i]; ++iCol) {
          if (iCol || i)
            outfile << ",";
          outfile << data[i]->at(iCol + iRow * _ColumnCount[i]);
        }
      }
      outfile << std::endl;
    }
  }

  outfile.close();
  newState->setOutputName(getFilename());
}

}

}
