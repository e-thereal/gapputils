/*
 * Filename.h
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_HOST_INPUTS_FILENAME_H_
#define GAPPUTLIS_HOST_INPUTS_FILENAME_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace interfaces {

namespace parameters {

class Filename : public gapputils::workflow::DefaultWorkflowElement<Filename> {

  InitReflectableClass(Filename)

  Property(Value, std::string)
  Property(Pattern, std::string)

private:
  static int patternId;

public:
  Filename();
  virtual ~Filename();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

namespace inputs {

class Filename : public gapputils::workflow::DefaultWorkflowElement<Filename> {

  InitReflectableClass(Filename)

  Property(Value, std::string)
  Property(Pattern, std::string)

private:
  static int patternId;

public:
  Filename();
  virtual ~Filename();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTLIS_HOST_INPUTS_FILENAME_H_ */
