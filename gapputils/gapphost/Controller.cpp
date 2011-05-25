/*
 * Controller.cpp
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#include "Controller.h"

#include <capputils/Xmlizer.h>

using namespace std;
using namespace capputils;

namespace gapputils {

namespace host {

Controller* Controller::instance = 0;

Controller::Controller() : model(DataModel::getInstance()) {
}

Controller::~Controller() {
}

Controller& Controller::getInstance() {
  if (!instance)
    instance = new Controller();
  return *instance;
}

/*void Controller::saveCurrentWorkflow(const std::string& filename) {
  //Xmlizer::ToFile(filename, model.getMainWorkflow()->getXml());
  Xmlizer::ToXml(filename, *model.getMainWorkflow());
}*/

}

}
