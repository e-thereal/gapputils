#pragma once

#ifndef _GAPPUTILS_CSVWRITER_H_
#define _GAPPUTILS_CSVWRITER_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class CsvWriter : public workflow::WorkflowElement
{
  InitReflectableClass(CsvWriter)

  Property(Filename, std::string)
  Property(ColumnCount, int)
  Property(RowCount, int)
  Property(Data, double*)

public:
  CsvWriter(void);
  virtual ~CsvWriter(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

}

#endif
