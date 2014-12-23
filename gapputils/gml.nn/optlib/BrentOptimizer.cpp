/*
 * BrentOptimizer.cpp
 *
 *  Created on: Nov 4, 2008
 *      Author: tombr
 */

#include "BrentOptimizer.h"

#include <cmath>
#include <iostream>
#include <sstream>

#include "NegationFunction.h"
#include "OptimizerException.h"

using namespace optlib;
using namespace std;

BrentOptimizer::BrentOptimizer() : ObservableLineOptimizer(),
    tolerance(1e-2), stepSize(1), checkMinLowerBound(false), checkMaxUpperBound(false)
{
}

string BrentOptimizer::getName() const {
  return "BrentOptimizer";
}

void BrentOptimizer::minimize(double& result, IFunction<DomainType>& function) {
  DomainType left, middle, right;

  fireEventTriggered(BrentStepChanged(BrentStepChanged::FindBracket));
  findBracket(left, middle, right, function);

  BrentState st;

  // Initialise state
  fireEventTriggered(BrentStepChanged(BrentStepChanged::Initialize));
  st.w = st.v = left + Gold * (right - left);
  st.f_w = st.f_v = function.eval(st.v);
  st.d = st.e = 0;
  st.middleValue = function.eval(middle);

  fireEventTriggered(BrentStepChanged(BrentStepChanged::Iterate));
  for (int i = 0; i < getMaxIterationCount(); ++i) {
    iterate(st, left, middle, right, function);
    fireEventTriggered(BrentIteration(left, middle, right));

    if (left > right) // left bracket is on the right side of right bracket
      THROW_OPTEX("ERROR_LINE_MINIMIZATION_FAILED");

    // Now check convergence
    double min_abs = min(fabs(left), fabs(right));

    if (left < 0 && right > 0)
      min_abs = 0;

    double toler = getTolerance() * (1.0 + min_abs);

    // Tolerance has been reached
    if ((right - left) < 4.0 * toler) {
      ostringstream strm;
      strm << "Brent iteration = " << i;
      fireEventTriggered(LineOptimizerLogEvent(strm.str()));
      fireEventTriggered(BrentStepChanged(BrentStepChanged::Finished));
      result = middle;
      return;
    }
  }
  fireEventTriggered(BrentStepChanged(BrentStepChanged::Finished));
  THROW_OPTEX("Maximum number of iterations reached in Brent minimizer.");
}

void BrentOptimizer::maximize(double& result, IFunction<DomainType>& function) {
  NegationFunction<DomainType> negfunc(&function);
  minimize(result, negfunc);
}

void BrentOptimizer::setParameter(int id, double value) {
  switch(id) {
  case LineTolerance: setTolerance(value); break;
  case LineMin:       minLowerBound = value; checkMinLowerBound = true; break;
  case LineMax:       maxUpperBound = value; checkMaxUpperBound = true; break;
  }
}

void BrentOptimizer::setParameter(int id, void* value) {
}

void BrentOptimizer::findBracket(DomainType& left, DomainType& middle, DomainType& right,
    IFunction<DomainType>& function)
{
  double leftValue, rightValue, middleValue;

  left = 0;
  right = 2 * getStepSize();
  leftValue = function.eval(left);
  rightValue = function.eval(right);

  // find a  middle point using the  golden ratio so  we can start
  // brent exponential search
  if (rightValue > leftValue) {
    middle = (right - left) * Gold + left;
    middleValue = function.eval(middle);
  } else {
    middle = (right - left) * (1.0 - Gold) + left;
    middleValue = function.eval(middle);
  }

  // find brackets
  for (int i = 0; i < getMaxIterationCount(); ++i) {
    if (fabs(left - middle) < getTolerance() && fabs(right - middle) < getTolerance()) {
      fireEventTriggered(LineOptimizerLogEvent("Reached a flat spot while bracketing."));
    }

    if (middleValue < leftValue) {
      if (middleValue < rightValue) {
        return;                                     // found good brackets
      } else if (middleValue > rightValue) {        // move all brackets to the right

        // If right bound has already been reached stop searching
        if (checkMaxUpperBound && right == maxUpperBound)
          return;

        left = middle;
        leftValue = middleValue;

        middle = right;
        middleValue = rightValue;

        right = (right - left) * ExpGold + right;   // Remark: left stores the
        if (checkMaxUpperBound && right > maxUpperBound)  
          right = maxUpperBound;                    // Limit right to maximum upper bound
        rightValue = function.eval(right);          //         old value of middle
      } else {                                      // fMiddle == fRight, shrink bracket
        right = middle;
        rightValue = middleValue;

        middle = (right - left) * Gold + left;
        middleValue = function.eval(middle);
      }
    } else {
      // This section was modified by Erick, because the one in
      // GSL is inappropriate for a bidirectional line search.

      // fMiddle >= fLeft, move brackets to the left

      // If left bound has already been reached stop searching
      if (checkMinLowerBound && left == minLowerBound)
        return;
      right = middle;
      rightValue = middleValue;

      middle = left;
      middleValue = leftValue;

      left = (left - right) * ExpGold + left;         // Remark: right stores the
      if (checkMinLowerBound && left < minLowerBound)
        left = minLowerBound;                         // Limit right to maximum upper bound
      leftValue = function.eval(left);                //         old middle value
    }

    if ((right - left) < getTolerance() * (right + left) * 0.5) {
      THROW_OPTEX("Brent bracketing error.");
    }
  }

  THROW_OPTEX("Brent bracketing error: maximum number of iterations reached.");
}

// written by Erick, transfered into a real line minimizer, so it handles one
// dimensional data (e.g. one double value)
void BrentOptimizer::iterate(BrentState &st, DomainType& left,
    DomainType& middle, DomainType& right, IFunction<DomainType>& function)
{

  double d = st.d;
  double e = st.e;
  const double v = st.v;
  const double w = st.w;
  const double f_v = st.f_v;
  const double f_w = st.f_w;
  const double f_z = st.middleValue;

  const double z = middle;
  const double w_lower = (z - left); // distance to boundaries
  const double w_upper = (right - z);
  const double toler = getTolerance() * (1.0 + fabs(z));

  double u = 0.0, f_u = 0.0;
  double p = 0.0, q = 0.0, r = 0.0;

  const double midpoint = 0.5 * (left + right);

  if (fabs(e) > toler) {

    // Fit parabola
    r = (z - w) * (f_z - f_v);
    q = (z - v) * (f_z - f_w);
    p = (z - v) * q - (z - w) * r;
    q = 2 * (q - r);

    // Formulas above  computed new_step =  p/q with a  wrong sign
    // (on purpose).  Correct this,  but in such  a way so  that q
    // would be positive.
    if (q > 0) {
      p = -p;
    } else {
      q = -q;
    }

    r = e;
    e = d;
  }

  if (fabs(p) < fabs(0.5 * q * r) && p < q * w_lower && p < q * w_upper) {
    double t2 = 2.0 * toler;
    //cout << "Perform parabolic step" << endl;

    d = p / q;
    u = z + d;

    if ((u - left) < t2 || (right - u) < t2) {  // changed right-z to right-u
      d = (z < midpoint) ? toler : -toler;
    }
  } else {
    //cout << "Perform golden section step" << endl;
    e = (z < midpoint) ? right - z : -(z - left);
    d = Gold * e;
  }

  if (fabs(d) >= toler)
    u = z + d;
  else
    u = z + ((d > 0) ? toler : -toler);

  st.e = e;
  st.d = d;

  f_u = function.eval(u);

  // One to one translation of the algol code in the Brent Book to C++
  if (f_u <= f_z) {
    if (u < z)
      right = z;
    else
      left = z;

    st.v = w, st.f_v = f_w, st.w = z, st.f_w = z, middle = u, st.middleValue = f_u;
  } else {
    if (u < z)
      left = u;
    else
      right = u;

    if (f_u <= f_w || w == z)
      st.v = w, st.f_v = f_w, st.w = u, st.f_w = f_u;
    else if (f_u <= f_v || v == z || v == w)
      st.v = u, st.f_v = f_u;
  }
}

double BrentOptimizer::getStepSize() const {
  return max(stepSize, 3*getTolerance());
}

double BrentOptimizer::getTolerance() const {
  return tolerance;
}

int BrentOptimizer::getMaxIterationCount() const {
  return 30;
}

void BrentOptimizer::setStepSize(double step) {
  stepSize = step;
}

void BrentOptimizer::setTolerance(double tol) {
  tolerance = tol;
}
