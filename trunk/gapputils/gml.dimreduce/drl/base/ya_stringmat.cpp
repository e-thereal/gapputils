/***************************************************************************
                               ya_stringmat.cpp
                             -------------------
                               W. Michael Brown

  Functions for dealing with matrices in strings
 
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

#ifndef YA_STRINGMAT_S_H
#define YA_STRINGMAT_S_H

YALANAMESPACE

// Set matrix equal to a matlab formatted string
template<class eltype>
template<class ya_type>
int ya_string_to_mat<eltype>::convert(const std::string &value, ya_type &mat) {
  int err_flag = 0;
  int ascii_flag = 0;
  std::string temp=value;

  // Transpose the matrix?
  bool transposeme=false;
  if (temp[temp.length()-1]=='\'') {
    temp[temp.length()-1]=' ';
    transposeme=true;
  }

  // Get the rows
  size_t tL=temp.length();
  for (size_t i=0; i<tL; i++)
    if (temp[i]=='[' || temp[i]==']' || temp[i]==',')
      temp[i]=' ';
  std::vector<std::string> row_str;
  ya_get_tokens(';',temp,row_str);
  if (row_str.empty()) {
    mat.clear();
    return err_flag;
  }
  ya_sizet row_size=row_str.size();
  if (ya_whitespace(row_str[row_size-1]))
    row_size--;
  if (row_size==0) {
    mat.clear();
    return err_flag;
  }
  
  // Get the columns
  for (ya_sizet i=0; i<row_size; i++) {
    std::vector<std::string> col_str;
    ya_get_tokens(row_str[i], col_str);
    if (col_str.empty()) {
      std::ostringstream owarn;
      owarn << "Error parsing matrix string:\n" << value << "\nat row token: " 
            << row_str[i];
      ya_addwarn(307,11,"YA_BaseT",owarn.str());
      err_flag = 307;
      if (i==0)
        return err_flag;
    }
    YA_DynT rvec;
    size_t cS=col_str.size();
    for (size_t j=0; j<cS; j++) {
      std::vector<std::string> e_tok;
      ya_get_tokens(':',col_str[j],e_tok);
      if (e_tok.size()>3) {
        std::ostringstream owarn;
        owarn << "Error parsing matrix string:\n" << value << "\nat token: " 
              << col_str[j];
        ya_addwarn(307,11,"YA_BaseT",owarn.str());
        err_flag=307;
        if (i==0)
          return err_flag;
      }
      eltype jump=1;
      ya_sizet high_index=1;
      if (e_tok.size()==3) {
        high_index=2;
        ascii_flag=ya_ascii_to(e_tok[1],jump);
      }
      if (e_tok.size()>1) {
        eltype one,two;
        ascii_flag=std::max(ascii_flag,ya_ascii_to(e_tok[0],one));
        ascii_flag=std::max(ascii_flag,ya_ascii_to(e_tok[high_index],two));
        double intp;
        if (modf(static_cast<double>(two-one)/double(jump),&intp) != 0)
          two=static_cast<eltype>(static_cast<double>(jump)*intp+one);
        if ( (one>two && static_cast<int>(jump)>0) || 
             (one<two && static_cast<int>(jump)<0) || jump==0) {
          std::ostringstream owarn;
          owarn << "Error parsing matrix string:\n" << value << "\nat token: " 
                << col_str[j];
          ya_addwarn(307,11,"YA_BaseT",owarn.str());
          err_flag=307;
          if (i==0)
            return err_flag;
        }
        for (eltype k=one; ; k+=jump) {
          rvec.push_back(k);
          if (k==two)
            break;
        }
      } else {
        eltype one;
        ascii_flag=std::max(ascii_flag,ya_ascii_to(e_tok[0],one));
        rvec.push_back(one);
      }
    }
    if (i==0)
      mat.setup(row_size,rvec.numel());
    else
      if (rvec.numel()!=mat.cols()) {
        std::ostringstream owarn;
        owarn << "Error parsing matrix string:\n" << value << "\nat token: " 
              << row_str[i] << "\nAll rows must have the same number of "
              << "columns";
        ya_addwarn(307,11,"YA_BaseT",owarn.str());
        err_flag=307;
      }
    mat(i,":")=rvec;
  }
  if (transposeme)
    ip_transpose(mat);
    
  return std::max(ascii_flag,err_flag);
}
  
}
#endif
