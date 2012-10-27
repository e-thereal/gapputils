#include "CsvWriter.h"

#include <capputils/InputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/Verifier.h>
#include <capputils/EventHandler.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/OutputAttribute.h>

#include <gapputils/LabelAttribute.h>
#include <gapputils/HideAttribute.h>

#include <fstream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(CsvWriter)

  ReflectableBase(workflow::DefaultWorkflowElement<CsvWriter>)

  WorkflowProperty(Data, Input("D"), NotNull<Type>())
  WorkflowProperty(Filename, Output("Csv"), Filename(), NotEqual<Type>(""))

EndPropertyDefinitions

CsvWriter::CsvWriter() {
  setLabel("CsvWriter");
}

void CsvWriter::update(workflow::IProgressMonitor* monitor) const {
  std::ofstream outfile(getFilename().c_str());

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

  outfile.close();
  newState->setFilename(getFilename());
}

}

}
