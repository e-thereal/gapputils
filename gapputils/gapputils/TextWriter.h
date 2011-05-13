#pragma once

#ifndef _GAPPUTILS_TEXTWRITER_H_
#define _GAPPUTILS_TEXTWRITER_H_

#include "gapputils.h"
#include "WorkflowElement.h"

namespace gapputils {

class TextWriter : public workflow::WorkflowElement
{
  InitReflectableClass(TextWriter)

  Property(Text, std::string)
  Property(Filename, std::string)

public:
  TextWriter();
  virtual ~TextWriter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
