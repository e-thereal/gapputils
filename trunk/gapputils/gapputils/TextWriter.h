#pragma once

#ifndef _GAPPUTILS_TEXTWRITER_H_
#define _GAPPUTILS_TEXTWRITER_H_

#include "gapputils.h"
#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class TextWriter : public capputils::reflection::ReflectableClass,
                   public capputils::ObservableClass
{
  InitReflectableClass(TextWriter)

  Property(Text, std::string)
  Property(Filename, std::string)

public:
  TextWriter();
  virtual ~TextWriter();

  void changeHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif