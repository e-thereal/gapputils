/*
 * Filenames.h
 *
 *  Created on: Jun 21, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_HOST_INPUTS_FILENAMES_H_
#define GAPPUTLIS_HOST_INPUTS_FILENAMES_H_

#include <gapputils/CollectionElement.h>

namespace interfaces {

namespace inputs {

class Filenames : public gapputils::workflow::CollectionElement {

  InitReflectableClass(Filenames)

  Property(Values,std::vector<std::string>)
  Property(Value, std::string)
  Property(Pattern, std::string)

private:
  static int filenamesId;
  static int patternId;

public:
  Filenames();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

namespace outputs {

class Filenames : public gapputils::workflow::CollectionElement {

  InitReflectableClass(Filenames)

  Property(Values,std::vector<std::string>)
  Property(Value, std::string)

private:
  static int filenamesId;

public:
  Filenames();
};

}

}

#endif /* GAPPUTLIS_HOST_INPUTS_FILENAME_H_ */
