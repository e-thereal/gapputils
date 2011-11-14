/*
 * distributions.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: tombr
 */

#include "distributions.h"

namespace gapputils {

namespace ml {

float createBernoulliSample::operator()(const float& x) const {
  return (float)(((float)rand() / (float)RAND_MAX) < x);
}

float createNormalSample::operator()(const float& x) const {
  return normal(eng) + x;
}

}

}
