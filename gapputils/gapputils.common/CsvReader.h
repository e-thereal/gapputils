#pragma once

#ifndef _GAPPUTILS_CSVREADER_H_
#define _GAPPUTILS_CSVREADER_H_

#include <capputils/Enumerators.h>
#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gapputils {

namespace common {

CapputilsEnumerator(CsvReadMode, Flat, Structured)

class CsvReader : public DefaultWorkflowElement<CsvReader>
{

InitReflectableClass(CsvReader)

Property(Filename, std::string)
Property(FirstColumn, int)
Property(LastColumn, int)
Property(FirstRow, int)
Property(LastRow, int)
Property(Delimiter, std::string)
Property(Mode, CsvReadMode)

Property(Data, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
Property(FlatData, boost::shared_ptr<std::vector<double> >)
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
