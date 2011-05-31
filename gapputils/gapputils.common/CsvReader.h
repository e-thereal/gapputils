#pragma once

#ifndef _GAPPUTILS_CSVREADER_H_
#define _GAPPUTILS_CSVREADER_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class CsvReader : public workflow::WorkflowElement
{

InitReflectableClass(CsvReader)

Property(Filename, std::string)
Property(FirstColumn, int)
Property(LastColumn, int)
Property(FirstRow, int)
Property(LastRow, int)
Property(ColumnCount, int)
Property(RowCount, int)
Property(Data, double*)

private:
  mutable CsvReader* data;

public:
  CsvReader();
  virtual ~CsvReader(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

}

#endif
