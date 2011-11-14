/*
 * distributions.h
 *
 *  Created on: Nov 10, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_DISTRIBUTIONS_H_
#define GAPPUTILS_ML_DISTRIBUTIONS_H_

#include <random>

namespace gapputils {

namespace ml {

/**
 * \brief Samples from a Bernoulli distribution given the mean
 */
struct createBernoulliSample {
  float operator()(const float& x) const;
};

/**
 * \brief Sampling from a normal distribution with mean \c x and variance 1
 */
struct createNormalSample {
  mutable std::ranlux64_base_01 eng;
  mutable std::normal_distribution<float> normal;

  float operator()(const float& x) const;
};

}

}

#endif /* GAPPUTILS_ML_DISTRIBUTIONS_H_ */
