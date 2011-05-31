#pragma once

#ifndef _GAPPUTILS_TEXTWRITER_H_
#define _GAPPUTILS_TEXTWRITER_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

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

}

#endif
