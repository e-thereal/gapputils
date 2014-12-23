/*
 * OptimizerException.cpp
 *
 *  Created on: Nov 4, 2008
 *      Author: tombr
 */

#include "OptimizerException.h"

using namespace std;
using namespace optlib;

OptimizerException::OptimizerException(const string& filename, int line,
    const string& description) : filename(filename), line(line), description(description)
{

}

OptimizerException::~OptimizerException() throw() {

}

string OptimizerException::getDescription() const {
  return description;
}

string OptimizerException::getFilename() const {
  return filename;
}

int OptimizerException::getLine() const {
  return line;
}

const char* OptimizerException::what() const throw() {
  return getDescription().c_str();
}

OPTLIB_API ostream& operator<<(ostream& os, const OptimizerException& ex)
{
  os << "OptimizerException in file '" << ex.getFilename();
  os << "' at line '" << ex.getLine() << "': " << ex.getDescription();
  return os;
}
