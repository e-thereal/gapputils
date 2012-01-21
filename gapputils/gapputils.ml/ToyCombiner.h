/*
 * ToyCombiner.h
 *
 *  Created on: Jan 20, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TOYCOMBINER_H_
#define GAPPUTILS_ML_TOYCOMBINER_H_

#include <gapputils/CombinerInterface.h>

namespace gapputils {

namespace ml {

class ToyCombiner : public gapputils::workflow::CombinerInterface
{
  InitReflectableClass(ToyCombiner)

  Property(InputNames, std::vector<std::string>)
  Property(OutputNames, std::vector<std::string>)
  Property(InputName, std::string)
  Property(OutputName, std::string)

private:
  static int inputId, outputId;

public:
  ToyCombiner();
  virtual ~ToyCombiner();
};

}

}

#endif /* GAPPUTILS_ML_TOYCOMBINER_H_ */
