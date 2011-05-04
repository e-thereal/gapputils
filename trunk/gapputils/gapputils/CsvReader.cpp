#include "CsvReader.h"

#include <fstream>
#include <sstream>
#include <cmath>

#include <ObserveAttribute.h>
#include <FileExists.h>
#include <FilenameAttribute.h>
#include <DescriptionAttribute.h>
#include <EventHandler.h>
#include <Verifier.h>
#include <VolatileAttribute.h>

#include "OutputAttribute.h"
#include "LabelAttribute.h"
#include "HideAttribute.h"

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

enum EventIds {
  LabelId, FilenameId, FirstColumnId, LastColumnId, FirstRowId, LastRowId, ColumnCountId, RowCountId, DataId
};

BeginPropertyDefinitions(CsvReader)

  DefineProperty(Label, Label(), Observe(LabelId))
  DefineProperty(Filename, FileExists(), Filename(), Observe(FilenameId))
  DefineProperty(FirstColumn, Observe(FirstColumnId), Description("Zero-based index of the first column"))
  DefineProperty(LastColumn, Observe(LastColumnId), Description("Zero-based index of the last column. A value of -1 indicates to read until the end."))
  DefineProperty(FirstRow, Observe(FirstRowId), Description("Zero-based index of the first row"))
  DefineProperty(LastRow, Observe(LastRowId), Description("Zero-based index of the last row. A value of -1 indicates to read until the end."))
  DefineProperty(ColumnCount, Observe(ColumnCountId), Output())
  DefineProperty(RowCount, Observe(RowCountId), Output())
  DefineProperty(Data, Observe(DataId), Output(), Hide(), Volatile())

EndPropertyDefinitions

CsvReader::CsvReader(void) : _Label("Reader"), _Filename(""), _FirstColumn(0), _LastColumn(-1), _FirstRow(1), _LastRow(-1), _ColumnCount(0), _RowCount(0), _Data(0)
{
  Changed.connect(capputils::EventHandler<CsvReader>(this, &CsvReader::changeEventHandler));
}


CsvReader::~CsvReader(void)
{
  if (_Data)
    delete _Data;
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

void CsvReader::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (FilenameId <= eventId && eventId <= LastRowId && capputils::Verifier::Valid(*this)) {
    // Read the CVS file

    const int firstRow = getFirstRow();
    const int lastRow = getLastRow();
    const int firstColumn = getFirstColumn();
    const int lastColumn = getLastColumn();

    ifstream csvfile(getFilename().c_str());
    string line;
    vector<vector<double>*> dataVector;
    int columnCount = 0;
    for (int rowIndex = 0; getline(csvfile, line); ++rowIndex) {
      if (firstRow <= rowIndex && (lastRow == -1 || rowIndex <= lastRow)) {
        
        double value;
        vector<double>* dataRow = new vector<double>();
        vector<string> tokens;

        tokenize(line, tokens, ",");
        for (int columnIndex = 0; columnIndex < tokens.size(); ++columnIndex) { 
          if (firstColumn <= columnIndex && (lastColumn == -1 || columnIndex <= lastColumn)) {
            stringstream stream(tokens[columnIndex]);
            stream >> value;
            dataRow->push_back(value);
          }
        }
        columnCount = max(columnCount, (int)dataRow->size());
        dataVector.push_back(dataRow);

      }
    }
    csvfile.close();

    // Wrap up the data
    if (getData())
      delete getData();

    double* data = new double[dataVector.size() * columnCount];
    for (int i = 0, k = 0; i < dataVector.size(); ++i) {
      vector<double>* dataRow = dataVector[i];
      for (int j = 0; j < dataRow->size(); ++j, ++k) {
        data[k] = dataRow->at(j);
      }
      k += columnCount - dataRow->size();
      delete dataRow;
    }

    setColumnCount(columnCount);
    setRowCount(dataVector.size());
    setData(data);
  }
}

}
