#pragma once

#ifndef GML_CSVREADER_H_
#define GML_CSVREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace core {

CapputilsEnumerator(CsvReadMode, Flat, Structured)

class CsvReader : public DefaultWorkflowElement<CsvReader> {

  InitReflectableClass(CsvReader)

  Property(Filename, std::string)
  Property(FirstColumn, int)
  Property(LastColumn, int)
  Property(FirstRow, int)
  Property(LastRow, int)
  Property(RowIdCount, int)
  Property(Delimiter, std::string)
  Property(Mode, CsvReadMode)
  Property(FastRead, bool)

  Property(Data, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(FlatData, boost::shared_ptr<std::vector<double> >)
  Property(RowIds, boost::shared_ptr<std::vector<std::string> >)
  Property(ColumnCount, int)
  Property(RowCount, int)

public:
  CsvReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif
