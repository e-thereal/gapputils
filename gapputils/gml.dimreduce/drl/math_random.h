/***************************************************************************
                                math_random.h
                              W. Michael Brown
                             -------------------

    Random number generators

 __________________________________________________________________________

    Part of the Math Library
 __________________________________________________________________________

    begin      : Wed May 2 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H

#include <cstdlib>
#include <limits.h>
#include <cmath>

//using namespace std;

YALANAMESPACE

/// Random number generator
/** Class for random number generation based on a specified uniform
  * integer random number generator **/
template<class uniform_type>
class MathRandom {
 public:
  typedef typename uniform_type::rtype rtype;  // unsigned integer type, at least 32 bits
  
  /// Seed with 1
  MathRandom() : normal_stored(false) { rng.seed(1); }
  ~MathRandom() {}
  
  /// Seed the random number generator
  inline void seed(const rtype seed) { rng.seed(seed); }
  /// Seed the random number generator with the time
  inline void seed_time() { rng.seed(rtype(time(0))); }
  
  /// Return a uniform integer random number >=0 and <=max_rand()
  inline rtype rand() { return rng.uniform(); }
  
  /// Return the maximum integer random number
  inline rtype rand_max() { return rng.uniform_max(); }
  
  /// Return a uniform random number >=0 and <=max
  inline rtype rand(const rtype max) { return rand()%(max+1); }

  /// Return a random number >=0 and <=1
  inline double rand01() { return static_cast<double>(rand())/
                                  static_cast<double>(rand_max()); }

  /// Return a random number >0 and <1
  inline double rand01_exc()
    { return static_cast<double>(rand()+0.5)/(rand_max()+1.0); }
  
  /// Return a random number from a normal (Gaussian) by Box-Muller method
  /** stddev*normal()+mean produces a distribution with the specified
    * standard deviation and mean **/
  inline double normal() {
    double r=sqrt(-2.0*log(1.0-rand01_exc()));
    double phi=2.0*PI*static_cast<double>(rand())/(rand_max()+1.0);
    return r * cos(phi);
  }

 private:
  uniform_type rng;              // Uniform random number generator
  bool normal_stored;            // Already have a normal random number stored
  double normal_stored_value;    // Value of next normal if stored
};

// ---------------------------- UNIFORM GENERATORS --------------------------

/// Uniform random number generator (uses native rand() function)
class MathCStdRNG {
 public:
  typedef int rtype;  // unsigned integer type, at least 32 bits
  inline void seed(const unsigned seed) { srand(seed); }
  inline int uniform() { return rand(); }
  inline int uniform_max() { return RAND_MAX; }
};

/// Mersenne Twister RNG
/** - Period is 2^19937-1
  * Based on code by Takuji Nishimura and Makoto Matsumoto **/
class MathMersenneTwister {
 public:
  typedef unsigned long rtype;  // unsigned integer type, at least 32 bits
  enum { N = 624 };              // length of state vector
  enum { SAVE = N + 1 };         // length of array for save()

  /// Seed with 1
  MathMersenneTwister() { seed(1); }
  
  /// Return an integer between 0 and 2^32-1
  rtype uniform() {
    if( left == 0 ) reload();
      --left;
    register rtype s1;
    s1 = *pNext++;
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9d2c5680UL;
    s1 ^= (s1 << 15) & 0xefc60000UL;
    return ( s1 ^ (s1 >> 18) );
  }
  
  /// Return maximum random integer
  rtype uniform_max() { return 4294967295UL; }
  
  /// Set the seed
  void seed( const rtype oneSeed ) {
    initialize(oneSeed);
    reload();
  }
  
protected:
  enum { M = 397 };  // period parameter
  rtype state[N];   // internal state
  rtype *pNext;     // next value to get from state
  int left;          // number of values left before reload needed
  
  void initialize( const rtype seedvalue ) {
    register rtype *s = state;
    register rtype *r = state;
    register int i = 1;
    *s++ = seedvalue & 0xffffffffUL;
    for( ; i < N; ++i ) {
      *s++ = ( 1812433253UL * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffUL;
      r++;
    }
  }
  
  void reload() {
    register rtype *p = state;
    register int i;
    for( i = N - M; i--; ++p )
      *p = twist( p[M], p[0], p[1] );
    for( i = M; --i; ++p )
      *p = twist( p[M-N], p[0], p[1] );
    *p = twist( p[M-N], p[0], state[0] );
    left = N, pNext = state;
  }
         
  rtype hiBit( const rtype& u ) const { return u & 0x80000000UL; }
  rtype loBit( const rtype& u ) const { return u & 0x00000001UL; }
  rtype loBits( const rtype& u ) const { return u & 0x7fffffffUL; }
  rtype mixBits( const rtype& u, const rtype& v ) const
    { return hiBit(u) | loBits(v); }
  rtype twist(const rtype& m, const rtype& s0, const rtype& s1 ) const
    { return m ^ (mixBits(s0,s1)>>1) ^ (-loBit(s1) & 0x9908b0dfUL); }
};

}
#endif
