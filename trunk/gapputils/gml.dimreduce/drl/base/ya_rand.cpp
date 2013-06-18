/***************************************************************************
                                 ya_rand.cpp
                             -------------------
                               W. Michael Brown

  Functions for random number generation
 
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_RAND_S_H
#define YA_RAND_S_H

YALANAMESPACE

// -- Loop helpers
template <class ref, class Rng> struct _ya_ufor_rand {
  static inline void op(ref r, Rng &rng) { r=rng.rand01(); }
};
template <class ref, class Rng> struct _ya_ufor_rande {
  static inline void op(ref r, Rng &rng) { r=rng.rand01_exc(); }
};
template <class ref, class Rng> struct _ya_ufor_randn {
  static inline void op(ref r, Rng &rng) { r=rng.normal(); }
};
template <class ref, class Rng> struct _ya_ufor_randi {
  static inline void op(ref r, const std::pair<const typename Rng::rtype, Rng *> &p)
    { r=p.second->rand(p.first); }
};

template<class eltype, class mat_type, class Rng>
void ip_rand(YA_BaseT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_rand<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_rand(const YA_WrapperT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_rand<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_rande(YA_BaseT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_rande<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_rande(const YA_WrapperT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_rande<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_randn(YA_BaseT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_randn<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_randn(const YA_WrapperT &mat, Rng &rng) {
  ya_no_order_loop<_ya_ufor_randn<typename mat_type::ref,Rng> >::lp(mat,rng);
}

template<class eltype, class mat_type, class Rng>
void ip_randi(YA_BaseT &mat, const typename Rng::rtype max, Rng &rng) {
  const std::pair<const typename Rng::rtype, Rng *> tpair(max,&rng);
  ya_no_order_loop<_ya_ufor_randi<typename mat_type::ref,Rng> >::lp(mat,tpair);
}

template<class eltype, class mat_type, class Rng>
void ip_randi(const YA_WrapperT &mat, const typename Rng::rtype max, Rng &rng) {
  ya_no_order_loop<_ya_ufor_randi<typename mat_type::ref,Rng> >::lp(mat,max,rng);
}

}
#endif
