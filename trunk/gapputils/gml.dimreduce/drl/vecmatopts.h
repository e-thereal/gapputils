/***************************************************************************
                                vecmatopts.h
                             -------------------
                               W. Michael Brown

  Various options for VecMat routines

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Oct 27 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATOPTS_H
#define VECMATOPTS_H

YALANAMESPACE

enum {
  VMEIGRRR,   ///< Relatively Robust Representations
  VMEIGPOWER, ///< Power method
  VMEIGDQ,    ///< Divide and Conquer
  VMEIGSIMPLE ///< Simple Driver
};

/// Options for eigen decomposition
/** Current methods are VMFULLEIG and VMPOWEREIG for power method **/
class EigenOptions {
 public:
  EigenOptions();

  /// Return the method
  ya_sizet method();
  /// Set the method
  void method(ya_sizet);
  /// Return a string with a list of methods string names
  /** String is formatted method1, method2, ..., and method_n **/
  std::string methods();
  /// Set the method based on its string name
  /** Generates fatal error if the string does not correspond to a method **/
  void method(const std::string);
  /// Return the string describing a method from its ya_sizet id
  std::string method_name(ya_sizet);
  
  /// Return the convergence criteria for iterative methods
  double tol();
  /// Set the convergence criteria for iterative methods (default is 0.999)
  void tol(double);
  /// Return the maximum number of iterations for iterative methods
  ya_sizet max_iters();
  /// Set the maximum number of iterations for iterative methods (100)
  void max_iters(ya_sizet);
  
  /// Number of eigen vectors to return for iterative methods (2)
  ya_sizet dim();
  /// Set the number of eigen vectors to return for iterative methods (2)
  void dim(ya_sizet);
 private:
  std::map<std::string,ya_sizet> method_list;
  std::map<ya_sizet,std::string> method_ids;
  ya_sizet method_value;
  double tol_value;
  ya_sizet max_iters_value;
  ya_sizet dim_value;
};

}
#endif
