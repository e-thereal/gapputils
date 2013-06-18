/***************************************************************************
                               vecmatopts.cpp
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

#ifndef VECMATOPTS_CPP
#define VECMATOPTS_CPP
 
#include "vecmatopts.h"

using namespace yala;

EigenOptions::EigenOptions() {
  method_list["DQ"]=VMEIGDQ;
  method_ids[VMEIGDQ]="DQ";
  method_list["SIMPLE"]=VMEIGSIMPLE;
  method_ids[VMEIGSIMPLE]="SIMPLE";
  method_list["RRR"]=VMEIGRRR;
  method_ids[VMEIGRRR]="RRR";
  method_list["POWER"]=VMEIGPOWER;
  method_ids[VMEIGPOWER]="POWER";

  dim_value=10;
  method_value=VMEIGRRR;
  tol_value=0;
  max_iters_value=10000;  // Maximum iterations for power method
}

std::string EigenOptions::methods() {
  if (method_list.empty())
    return "";
  if (method_list.size()==1)
    return method_list.begin()->first;
  std::map<std::string,ya_sizet>::reverse_iterator rm=method_list.rbegin();
  std::string list="and "+rm->first;
  rm++;
  while (rm!=method_list.rend()) {
    list=rm->first+", "+list;
    rm++;
  }
  return list;
}

void EigenOptions::method(std::string methstr) {
  std::map<std::string,ya_sizet>::iterator m=method_list.find(methstr);
  if (m==method_list.end()) {
    std::ostringstream owarn;
    owarn << methstr << " is not a current eigen method. Options are " 
          << methods();
    ya_addwarn(0,999999999,"EigenOptions",owarn.str());
  }
  method_value=m->second;
}

ya_sizet EigenOptions::method() {
  return method_value;
}

// Return the string describing a method from its ya_sizet id
void EigenOptions::method(ya_sizet methid) {
  method_value=methid;
}

// Return the string describing a method from its ya_sizet id
std::string EigenOptions::method_name(ya_sizet methid) {
  return method_ids[methid];
}

ya_sizet EigenOptions::dim() {
  return dim_value;
}

void EigenOptions::dim(ya_sizet dimv) {
  dim_value=dimv;
}

double EigenOptions::tol() {
  return tol_value;
}

void EigenOptions::tol(double pe_epsv) {
  tol_value=pe_epsv;
}

ya_sizet EigenOptions::max_iters() {
  return max_iters_value;
}

void EigenOptions::max_iters(ya_sizet max_itersv) {
  max_iters_value=max_itersv;
}

#endif
