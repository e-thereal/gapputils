#include "CsvWriter.h"

#include "InputAttribute.h"
#include <ObserveAttribute.h>
#include "LabelAttribute.h"
#include <FilenameAttribute.h>
#include <NotEqualAssertion.h>
#include <Verifier.h>
#include <EventHandler.h>
#include "HideAttribute.h"
#include <VolatileAttribute.h>
#include "ShortNameAttribute.h"

#include <fstream>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(CsvWriter)

  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(Filename, Observe(PROPERTY_ID), Filename(), NotEqual<string>(""))
  DefineProperty(ColumnCount, ShortName("CC"), Observe(PROPERTY_ID), Input(), Volatile())
  DefineProperty(RowCount, ShortName("RC"), Observe(PROPERTY_ID), Input(), Volatile())
  DefineProperty(Data, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile())

EndPropertyDefinitions

CsvWriter::CsvWriter(void) : _Filename(""),
    _ColumnCount(0), _RowCount(0), _Data(0)
{
  setLabel("Writer");
}


CsvWriter::~CsvWriter(void)
{
}

void CsvWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!capputils::Verifier::Valid(*this))
    return;

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
    monitor->reportProgress(100 * i / rows);
  }

  outfile.close();
}

void CsvWriter::writeResults() {
}

}
