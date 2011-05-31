#pragma once

#ifndef _GAPPUTILS_PDFLATEX_H_
#define _GAPPUTILS_PDFLATEX_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class PdfLatex : public workflow::WorkflowElement
{

InitReflectableClass(PdfLatex)

Property(TexFilename, std::string)
Property(CommandName, std::string)
Property(ParameterString, std::string)
Property(OutputName, std::string)
Property(CommandOutput, std::string)

private:
  mutable PdfLatex* data;
  static int texId;

public:
  PdfLatex(void);
  virtual ~PdfLatex(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
