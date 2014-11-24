#include "CsvReader.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(CsvReader)

  ReflectableBase(workflow::DefaultWorkflowElement<CsvReader>)

  WorkflowProperty(Filename, Input("File"), Filename("CSV file (*.csv)"), FileExists())
  WorkflowProperty(FirstColumn, Description("Zero-based index of the first column"))
  WorkflowProperty(LastColumn, Description("Zero-based index of the last column. A value of -1 indicates to read until the end."))
  WorkflowProperty(FirstRow, Description("Zero-based index of the first row"))
  WorkflowProperty(LastRow, Description("Zero-based index of the last row. A value of -1 indicates to read until the end."))
  WorkflowProperty(RowIdCount, Description("If greater than 0, the first RowIdCount columns are assumed to contain the row ID."))
  WorkflowProperty(Delimiter)
  WorkflowProperty(Mode, Enumerator<Type>())
  WorkflowProperty(FastRead, Flag())
  
  WorkflowProperty(Data, Output("D"))
  WorkflowProperty(FlatData, Output("FD"))
  WorkflowProperty(RowIds, Output("Ids"))
  WorkflowProperty(Header, NoParameter())
  WorkflowProperty(ColumnCount, NoParameter())
  WorkflowProperty(RowCount, NoParameter())

EndPropertyDefinitions

CsvReader::CsvReader()
 : _FirstColumn(0), _LastColumn(-1), _FirstRow(0), _LastRow(-1), _RowIdCount(0),
   _Delimiter(","), _Mode(CsvReadMode::Structured), _FastRead(false), _ColumnCount(0), _RowCount(0)
{
  setLabel("CsvReader");
}

void tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ")
{
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
      // Found a token, add it to the vector.
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
  }
}

void CsvReader::update(IProgressMonitor* monitor) const {
  using namespace std;

  // Read the CVS file
  const int firstRow = getFirstRow();
  const int lastRow = getLastRow();
  const int firstColumn = getFirstColumn();
  const int lastColumn = getLastColumn();

  ifstream csvfile(getFilename().c_str());
  csvfile.seekg (0, ios::end);
  int fileSize = csvfile.tellg();
  csvfile.seekg (0, ios::beg);
  string line;

  boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > data(
      new std::vector<boost::shared_ptr<std::vector<double> > >());

  boost::shared_ptr<std::vector<std::string> > rowIds;

  if (getRowIdCount() > 0) {
    rowIds = boost::make_shared<std::vector<std::string> >();
  }

  int columnCount = 0;
  const bool fast = getFastRead();
  for (int rowIndex = 0; getline(csvfile, line); ++rowIndex) {
    if (rowIndex == 0)
      newState->setHeader(line);
    if (firstRow <= rowIndex && (lastRow == -1 || rowIndex <= lastRow)) {

      double value;
      boost::shared_ptr<vector<double> > dataRow(new vector<double>());
      vector<string> tokens;

      tokenize(line, tokens, getDelimiter());
      if (rowIds) {
        stringstream idStream;
        for (int iCol = 0; iCol < getRowIdCount(); ++iCol)
          idStream << (iCol ? "," : "") << tokens[iCol];
        rowIds->push_back(idStream.str());
      }
      for (int iCol = firstColumn; iCol < (int)tokens.size() && (lastColumn == -1 || iCol <= lastColumn); ++iCol) {
        stringstream stream(tokens[iCol]);
        stream >> value;
        dataRow->push_back(value);
      }
      columnCount = max(columnCount, (int)dataRow->size());
      data->push_back(dataRow);
      if (!fast)
        monitor->reportProgress(100 * csvfile.tellg() / fileSize);
    }
  }
  csvfile.close();

  newState->setRowIds(rowIds);
  newState->setColumnCount(columnCount);
  newState->setRowCount(data->size());

  if (getMode() == CsvReadMode::Structured) {
    newState->setData(data);
  } else {
    boost::shared_ptr<std::vector<double> > flatData(new std::vector<double>());
    for (size_t iRow = 0; iRow < data->size(); ++iRow) {
      std::vector<double>& row = *data->at(iRow);
      for (int iCol = 0; iCol < columnCount; ++iCol) {
        if (iCol < (int)row.size())
          flatData->push_back(row[iCol]);
        else
          flatData->push_back(0.0);
      }
    }
    newState->setFlatData(flatData);
  }
}

}

}
