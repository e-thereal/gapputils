#pragma once

#ifndef _GAPPUTILS_PDFLATEX_H_
#define _GAPPUTILS_PDFLATEX_H_

#include "gapputils.h"
#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class PdfLatex : public capputils::reflection::ReflectableClass,
                 public capputils::ObservableClass
{

InitReflectableClass(PdfLatex)

Property(TexFilename, std::string)
Property(CommandName, std::string)
Property(ParameterString, std::string)
Property(OutputName, std::string)
Property(CommandOutput, std::string)

public:
  PdfLatex(void);
  virtual ~PdfLatex(void);

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif