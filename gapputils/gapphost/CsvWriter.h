#pragma once

#ifndef _GAPPUTILS_CSVWRITER_H_
#define _GAPPUTILS_CSVWRITER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class CsvWriter : public capputils::reflection::ReflectableClass,
                  public capputils::ObservableClass
{
  InitReflectableClass(CsvWriter)

  Property(Label, std::string)
  Property(Filename, std::string)
  Property(ColumnCount, int)
  Property(RowCount, int)
  Property(Data, double*)

public:
  CsvWriter(void);
  virtual ~CsvWriter(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif