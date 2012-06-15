#pragma once
#ifndef _GAPPUTILS_ALTOVAXML_H_
#define _GAPPUTILS_ALTOVAXML_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace host {

namespace internal {

class XslTransformation : public workflow::WorkflowElement
{
  InitReflectableClass(XslTransformation)

  Property(InputName, std::string)
  Property(OutputName, std::string)
  Property(XsltName, std::string)
  Property(CommandName, std::string)
  Property(CommandOutput, std::string)
  Property(OutputExtension, std::string)
  Property(InSwitch, std::string)
  Property(OutSwitch, std::string)
  Property(XsltSwitch, std::string)

private:
  mutable XslTransformation* data;
  static int inputId, extId;

public:
  XslTransformation(void);
  virtual ~XslTransformation(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

}

#endif
