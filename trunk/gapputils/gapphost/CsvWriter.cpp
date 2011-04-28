#include "CsvWriter.h"

#include <InputAttribute.h>
#include <ObserveAttribute.h>
#include <LabelAttribute.h>
#include <FilenameAttribute.h>
#include <NotEqualAssertion.h>
#include <Verifier.h>
#include <EventHandler.h>
#include <HideAttribute.h>

#include <fstream>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

enum PropertyIds {
  LabelId, FilenameId, ColumnCountId, RowCountId, DataId
};

BeginPropertyDefinitions(CsvWriter)

  DefineProperty(Label, Observe(LabelId), Label())
  DefineProperty(Filename, Observe(FilenameId), Filename(), NotEqual<string>(""))
  DefineProperty(ColumnCount, Observe(ColumnCountId), Input())
  DefineProperty(RowCount, Observe(RowCountId), Input())
  DefineProperty(Data, Observe(DataId), Input(), NotEqual<double*>(0), Hide())

EndPropertyDefinitions

CsvWriter::CsvWriter(void) : _Label("Writer"), _Filename(""), _ColumnCount(0), _RowCount(0), _Data(0)
{
  Changed.connect(capputils::EventHandler<CsvWriter>(this, &CsvWriter::changeEventHandler));
}


CsvWriter::~CsvWriter(void)
{
}

void CsvWriter::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (!capputils::Verifier::Valid(*this))
    return;

  if (eventId == DataId || eventId == FilenameId) {
    ofstream outfile(getFilename().c_str());

    double* data = getData();
    int rows = getRowCount();
    int cols = getColumnCount();

    for (int i = 0, k = 0; i < rows; ++i) {
      if (cols)
        outfile << data[k++];
      for (int j = 1; j < cols; ++j, ++k)
        outfile << ", " << data[k];
      outfile << endl;
    }

    outfile.close();
  }
}

}
