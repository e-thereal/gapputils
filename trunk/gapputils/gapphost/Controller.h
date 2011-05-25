/*
 * Controller.h
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#include "DataModel.h"

namespace gapputils {

namespace host {

class Controller {

private:
  static Controller* instance;
  DataModel& model;

protected:
  Controller();

public:
  virtual ~Controller();

  static Controller& getInstance();

  //void saveCurrentWorkflow(const std::string& filename);
};

}

}

#endif /* CONTROLLER_H_ */
