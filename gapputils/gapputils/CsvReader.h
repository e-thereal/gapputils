#pragma once

#ifndef _GAPPUTILS_CSVREADER_H_
#define _GAPPUTILS_CSVREADER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class CsvReader : public capputils::reflection::ReflectableClass,
                  public capputils::ObservableClass
{

InitReflectableClass(CsvReader)

Property(Label, std::string)
Property(Filename, std::string)
Property(FirstColumn, int)
Property(LastColumn, int)
Property(FirstRow, int)
Property(LastRow, int)
Property(ColumnCount, int)
Property(RowCount, int)
Property(Data, double*)

public:
  CsvReader(void);
  virtual ~CsvReader(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif
