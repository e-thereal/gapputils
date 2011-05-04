#pragma once
#ifndef _GAPPUTILS_ALTOVAXML_H_
#define _GAPPUTILS_ALTOVAXML_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class AltovaXml : public capputils::reflection::ReflectableClass,
                  public capputils::ObservableClass
{
  InitReflectableClass(AltovaXml)

  Property(InputName, std::string)
  Property(OutputName, std::string)
  Property(XsltName, std::string)
  Property(CommandName, std::string)
  Property(CommandOutput, std::string)

public:
  AltovaXml(void);
  virtual ~AltovaXml(void);

  void changeHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif
