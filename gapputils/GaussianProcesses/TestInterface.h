/*
 * TestInterface.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef TESTINTERFACE_H_
#define TESTINTERFACE_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>

namespace GaussianProcesses {

class TestInterface : public capputils::reflection::ReflectableClass,
                      public capputils::ObservableClass
{
  InitReflectableClass(TestInterface)

  Property(Pdf, std::string)

public:
  TestInterface();
  virtual ~TestInterface();
};

}

#endif /* TESTINTERFACE_H_ */
