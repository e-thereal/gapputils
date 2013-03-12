/*
 * String.h
 *
 *  Created on: Jun 8, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_STRING_H_
#define GAPPUTILS_HOST_STRING_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace host {

namespace inputs {

class String : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(String)

  Property(Value, std::string)

private:
  mutable String* data;

public:
  String();
  virtual ~String();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

namespace outputs {

class String : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(String)

  Property(Value, std::string)

private:
  mutable String* data;

public:
  String();
  virtual ~String();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

}

}

#endif /* GAPPUTILS_HOST_STRING_H_ */
