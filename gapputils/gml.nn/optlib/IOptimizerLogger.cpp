#include "IOptimizerLogger.h"

using namespace std;
using namespace optlib;

OPTLIB_API ostream& operator<<(ostream& os, const IOptimizerLogger& logger)
{
  logger.print(os);
  return os;
}
