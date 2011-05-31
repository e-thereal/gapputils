#include "CsvReader.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <capputils/ObserveAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/DescriptionAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/LabelAttribute.h>
#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(CsvReader)

  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(FirstColumn, ShortName("FC"), Observe(PROPERTY_ID), Description("Zero-based index of the first column"), Input(), TimeStamp(PROPERTY_ID))
  DefineProperty(LastColumn, ShortName("LC"), Observe(PROPERTY_ID), Description("Zero-based index of the last column. A value of -1 indicates to read until the end."), Input(), TimeStamp(PROPERTY_ID))
  DefineProperty(FirstRow, ShortName("FR"), Observe(PROPERTY_ID), Description("Zero-based index of the first row"), Input(), TimeStamp(PROPERTY_ID))
  DefineProperty(LastRow, ShortName("LR"), Observe(PROPERTY_ID), Description("Zero-based index of the last row. A value of -1 indicates to read until the end."), Input(), TimeStamp(PROPERTY_ID))
  DefineProperty(Filename, ShortName("File"), Observe(PROPERTY_ID), FileExists(), Filename(), Input(), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, ShortName("CC"), Observe(PROPERTY_ID), Output(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(RowCount, ShortName("RC"), Observe(PROPERTY_ID), Output(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Data, Observe(PROPERTY_ID), Output(), Hide(), Volatile(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

CsvReader::CsvReader() : _Filename(""), _FirstColumn(0), _LastColumn(-1),
_FirstRow(1), _LastRow(-1), _ColumnCount(0), _RowCount(0), _Data(0), data(0)
{
  setLabel("Reader");
}


CsvReader::~CsvReader(void)
{
  if (_Data)
    delete _Data;

  if (data)
    delete data;
}

void tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ")
{
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)
  {
      // Found a token, add it to the vector.
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
  }
}

void CsvReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new CsvReader();

  if (!capputils::Verifier::Valid(*this))
    return;

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
  vector<vector<double>*> dataVector;
  int columnCount = 0;
  for (int rowIndex = 0; getline(csvfile, line); ++rowIndex) {
    if (firstRow <= rowIndex && (lastRow == -1 || rowIndex <= lastRow)) {

      double value;
      vector<double>* dataRow = new vector<double>();
      vector<string> tokens;

      tokenize(line, tokens, ",");
      for (int columnIndex = 0; columnIndex < (int)tokens.size(); ++columnIndex) {
        if (firstColumn <= columnIndex && (lastColumn == -1 || columnIndex <= lastColumn)) {
          stringstream stream(tokens[columnIndex]);
          stream >> value;
          dataRow->push_back(value);
        }
      }
      columnCount = max(columnCount, (int)dataRow->size());
      dataVector.push_back(dataRow);
      monitor->reportProgress(100 * csvfile.tellg() / fileSize);
    }
  }
  csvfile.close();

  // Wrap up the data
  if (this->data->getData())
    delete this->data->getData();

  double* data = new double[dataVector.size() * columnCount];
  for (unsigned i = 0, k = 0; i < dataVector.size(); ++i) {
    vector<double>* dataRow = dataVector[i];
    for (unsigned j = 0; j < dataRow->size(); ++j, ++k) {
      data[k] = dataRow->at(j);
    }
    k += columnCount - dataRow->size();
    delete dataRow;
  }

  this->data->setColumnCount(columnCount);
  this->data->setRowCount(dataVector.size());
  this->data->setData(data);
}

void CsvReader::writeResults() {
  if (getData())
    delete getData();

  size_t count = data->getColumnCount() * data->getRowCount();
  double* _data = new double[count];
  std::copy(data->getData(), data->getData() + count, _data);

  setColumnCount(data->getColumnCount());
  setRowCount(data->getRowCount());
  setData(_data);
}

}

}
