/***************************************************************************
                                  ya_io.cpp
                             -------------------
                               W. Michael Brown

  Functions for matrix input/output
 
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

#ifndef YA_IO_S_H
#define YA_IO_S_H

YALANAMESPACE

template<class eltype, class mat_type>
std::ostream & operator<<(std::ostream &out, const YA_BaseT &vm) {
  const ya_sizet nR=vm.rows();
  const ya_sizet nC=vm.cols();
  for (ya_sizet i=0; i<nR; i++) {
    for (ya_sizet j=0; j<nC; j++) {
      out << vm(i,j) << " ";
    }
    if (i!=nR-1)
      out << std::endl;
  }
  return out;
}

int ya_fileopen(std::ifstream &in, const char *filename) {
  in.clear();
  in.open(filename);
  if (!in) {
    ya_addwarn(1,15,"YA_IO",
               "Could not open "+std::string(filename)+" for input!");
    return 1;
  }
  return 0;
}

int ya_fileopen(std::ifstream &in, const std::string &filename) {
  in.clear();
  in.open(filename.c_str());
  if (!in) {
    ya_addwarn(1,15,"YA_IO",
               "Could not open "+std::string(filename)+" for input!");
    return 1;
  }
  return 0;
}

int ya_fileopenbinary(std::ifstream &in, const std::string &filename) {
  in.clear();
  in.open(filename.c_str(), std::ios::binary);
  if (!in) {
    ya_addwarn(1,15,"YA_IO",
               "Could not open "+std::string(filename)+" for input!");
    return 1;
  }
  return 0;
}

int ya_fileopen(std::ofstream &out, const char *filename) {
  out.clear();
  out.open(filename);
  if (!out) {
    ya_addwarn(2,15,"YA_IO",
               "Could not open "+std::string(filename)+" for output!");
    return 2;
  }
  return 0;
}

int ya_fileopen(std::ofstream &out, const std::string &filename) {
  out.clear();
  out.open(filename.c_str());
  if (!out) {
    ya_addwarn(2,15,"YA_IO",
               "Could not open "+std::string(filename)+" for output!");
    return 2;
  }
  return 0;
}

int ya_fileopenbinary(std::ofstream &out, const std::string &filename) {
  out.clear();
  out.open(filename.c_str(),std::ios::binary);
  if (!out) {
    ya_addwarn(2,15,"YA_IO",
               "Could not open "+std::string(filename)+" for output!");
    return 2;
  }
  return 0;
}

int ya_fileopenapp(std::ofstream &out, const std::string &filename) {
  out.clear();
  out.open(filename.c_str(),std::ios::app);
  if (!out) {
    ya_addwarn(2,15,"YA_IO",
               "Could not open "+std::string(filename)+" for output!");
    return 2;
  }
  return 0;
}

int ya_fileclose(std::ifstream &in) {
  in.close();
  return 0;
}

int ya_fileclose(std::ofstream &out) {
  if (out.fail()) {
    ya_addwarn(10,15,"YA_IO","Error writing to output file!");
    return 10;
  }
  out.close();
  return 0;
}

// ----------------- I/O Helper Functions

template <int mtype, int dtype> struct _ya_binary_helper;
template <int mtype,int diag> struct _ya_write_helper;
template <class eltype, int mtype> struct _ya_readsparse_helper;

// ----------------- I/O Helper Functions (Binary)
// Can't write a binary file (pointers, etc)
template <int mtype> 
struct _ya_binary_helper<mtype,0> {
  template<class eltype, class mat_type>
  static void write(std::ostream &out, const YA_BaseT &mat)
    { write(out,mat,YA_DEFAULT_IO); }
  template<class eltype, class mat_type>
  static int read(std::istream &in, YA_BaseT &mat)
    { return read(in,mat,YA_DEFAULT_IO); }
  template<class eltype, class mat_type>
  static int read(std::istream &in, const YA_WrapperT &mat)
    { return read(in,mat,YA_DEFAULT_IO); }
};
// This one writes
template<> struct _ya_binary_helper<1,1> {
  template<class eltype, class mat_type>
  static void write(std::ostream &out, const YA_BaseT &mat) {
    out << "YA_Binary " << mat.rows() << " " << mat.cols() << " "
        << YA_TRAIT::VME_MEMORY << std::endl;
    out.write((char *)mat.c_ptr(),sizeof(eltype)*mat.numel());
    out << std::endl;
  }
  template<class eltype, class mat_type>
  static int read(std::istream &in, YA_BaseT &mat) {
    ya_sizet R,C;
    int order;
    char space;
    in >> R >> C >> order;
    if (order!=YA_TRAIT::VME_ORDER)
      mat.setup(C,R);
    else
      mat.setup(R,C);
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      mat.clear();
      return 301;
    }
    in.get(space);
    in.read((char*)mat.c_ptr(),sizeof(eltype)*mat.numel());
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      mat.clear();
      return 301;
    }
    if (order!=YA_TRAIT::VME_ORDER)
      ip_transpose(mat);
    return 0;
  }
  // Const copy
  template<class eltype, class mat_type>
  static int read(std::istream &in, const YA_WrapperT &mat) {
    ya_sizet R,C;
    int order;
    char space;
    in >> R >> C >> order;
    if (order!=YA_TRAIT::VME_ORDER)
      mat.setup(C,R);
    else
      mat.setup(R,C);
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      mat.clear();
      return 301;
    }
    in.get(space);
    in.read((char*)mat.c_ptr(),sizeof(eltype)*mat.numel());
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      mat.clear();
      return 301;
    }
    if (order!=YA_TRAIT::VME_ORDER)
      ip_transpose(mat);
    return 0;
  }
};
// Can't write a binary file directly, but create temporary matrix
template <int mtype> 
struct _ya_binary_helper<mtype,1> {
  template<class eltype, class mat_type>
  static void write(std::ostream &out, const YA_BaseT &mat)
    { _ya_binary_helper<1,1>::write(out,YA_MatT(mat)); }
  template<class eltype, class mat_type>
  static int read(std::istream &in, YA_BaseT &mat) {
    int err_flag;
    if (YA_TRAIT::VME_ORDER==1) {
      YA_MatRMT t;
      err_flag=_ya_binary_helper<1,1>::read(in,t);
      mat=t;
    } else {
      YA_MatT t;
      err_flag=_ya_binary_helper<1,1>::read(in,t);
      mat=t;
    }
    return err_flag;
  }
  // Const copy
  template<class eltype, class mat_type>
  static int read(std::istream &in, const YA_WrapperT &mat) {
    int err_flag;
    if (YA_TRAIT::VME_ORDER) {
      YA_MatRMT t;
      err_flag=_ya_binary_helper<1,1>::read(in,t);
      mat=t;
    } else {
      YA_MatT t;
      err_flag=_ya_binary_helper<1,1>::read(in,t);
      mat=t;
    }
    return err_flag;
  }
};
     

// ----------------- I/O Helper Functions (Symmetric)

template<> struct _ya_write_helper<2,0> {
  template<class ya_type>
  static void write(std::ostream &out,const ya_type &mat) {
    out << "VecMatSym " << mat.rows() << std::endl;
    const ya_sizet nE=mat.numel();
    for (ya_sizet i=0; i<nE; i++)
      out << mat(i) << std::endl;
  }
};

template<class eltype, class mat_type, class ya_type>
int _ya_readsym_helper(std::istream &in, ya_type &mat) {
  int err_flag=0;

  ya_sizet nR;
  in >> nR;
  if (in.eof() || !in) {
    ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
    mat.clear();
    return 301;
  }
  if (YA_TRAIT::VME_MEMORY!=2) {
    VecMatSym<eltype> tmat(nR,nR);
    eltype value;
    const ya_sizet nE=tmat.numel();
    for (ya_sizet i=0; i<nE; i++) {
      in >> value;
      tmat(i)=value;
    }
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      err_flag = 301;
    }
    mat=tmat;
  } else {
    mat.setup(nR,nR);
    eltype value;
    const ya_sizet nE=mat.numel();
    for (ya_sizet i=0; i<nE; i++) {
      in >> value;
      mat(i)=value;
    }
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      err_flag = 301;
    }
  }
  
  return err_flag;
}

// ----------------- I/O Helper Functions (Sparse)

template<> struct _ya_write_helper<4,0> {
  template<class ya_type>
  static void write(std::ostream &out,const ya_type &mat) {
    out << "VecMatSparse " << mat.rows() << " " << mat.cols() << std::endl;
    mat.stv().write_nonzero(out);
  }
};

template<class eltype> 
struct _ya_readsparse_helper<eltype,4> {
  template<class ya_type>
  static int read(std::istream &in,ya_type &mat) {
    mat.clear();
    ya_sizet nR, nC;
    in >> nR >> nC;
    if (in.eof() || !in) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      return 301;
    }
    mat.setup(nR,nC);
    return mat.read_nonzero(in);
  }
};
  
template<class eltype,int mtype> 
struct _ya_readsparse_helper {
  template<class ya_type>
  static int read(std::istream &in,ya_type &mat) {
    VecMatSparse<eltype> tmat;
    int err_flag=_ya_readsparse_helper<eltype,4>::read(in,tmat);    
    mat=tmat;
    return err_flag;
  }
};

// ----------------- I/O Helper Functions (Diagonal)

template<int mtype> struct _ya_write_helper<mtype,1> {
  template<class ya_type>
  static void write(std::ostream &out,const ya_type &mat) {
    out << "VecMatDiag " << mat.rows() << std::endl;
    const ya_sizet nR=mat.rows();
    for (ya_sizet i=0; i<nR; i++)
      out << mat(i,i) << std::endl;
  }
};

// Read in a matrix
template<class eltype, class mat_type, class ya_type>
int _ya_readdiag_helper(std::istream &in, ya_type &mat) {
  ya_sizet nR;
  in >> nR;
  if (in.eof() || !in) {
    ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
    mat.clear();
    return 301;
  }
  mat.setup(nR,nR);
  if (YA_TRAIT::VME_DIAGONAL==0)
    zero(mat);
  eltype value;
  for (ya_sizet i=0; i<nR; i++) {
    in >> value;
    mat(i,i)=value;
  }
  if (in.eof() || !in) {
    ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
    return 301;
  }
  return 0;
}
  
// ----------------- I/O Helper Functions (Regular)

template<int mtype, int diag> struct _ya_write_helper {
  template<class ya_type>
  static void write(std::ostream &out,const ya_type &mat) {
    out << "VecMat " << mat.rows() << " " << mat.cols() << std::endl;
    const ya_sizet nE=mat.numel();
    for (ya_sizet i=0; i<nE; i++)
      out << mat(i) << std::endl;
  }
};

// Read in a matrix
template<class eltype, class mat_type, class ya_type>
int _ya_read_helper(std::istream &in, ya_type &mat) {
  ya_sizet nR, nC;
  in >> nR >> nC;
  if (in.eof() || !in) {
    ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
    mat.clear();
    return 301;
  }
  int err_flag = 0;
  if (YA_TRAIT::VME_MEMORY==2) {
    if (nR!=nC) {
      ya_addwarn(301,19,"YA_IO",
                 "Attempt to read non-square matrix into symmteric type.");
      err_flag = 301;
    }
    YA_MatT tmat(nR,nC);
    eltype value;
    const ya_sizet nE=nR*nC;
    for (ya_sizet i=0; i<nE; i++) {
      in >> value;
      tmat(i)=value;
    }
    mat=tmat;
  } else {
    mat.setup(nR,nC);
    eltype value;
    const ya_sizet nE=mat.numel();
    for (ya_sizet i=0; i<nE; i++) {
      in >> value;
      mat(i)=value;
    }
  }
  if (in.eof() || !in) {
    ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
    err_flag = 301;
  }
  return err_flag;
}

// ----------------- I/O Helper Functions (ASCII)

// Read in a MATLAB formatted text file and store as the transpose
template<class eltype>
int _ya_readascii_helper(std::istream &in,YA_DynT &mat) {
  mat.clear();
  char line[200000];
  ya_sizet row_num=0;
  ya_sizet number_cols=0;
  int err_flag = 0;

  while (1) {
    in.getline(line,200000);
    std::vector<std::string> tokens;
    ya_get_tokens(std::string(line),tokens);
    if (tokens.size()==0)
      break;
    if (row_num==0)
      number_cols=tokens.size();
    else if (number_cols!=static_cast<ya_sizet>(tokens.size())) {
      ya_addwarn(301,19,"YA_IO","Improperly formatted YALA++ file.");
      err_flag = 301;
      break;
    }
    std::istringstream rowin(line);
    size_t nT=tokens.size();
    for (size_t i=0; i<nT; i++) {
      eltype value;
      rowin >> value;
      mat.push_back(value);
    }
    row_num++;
  }

  // Resize the matrix
  mat.reshape(number_cols,row_num);
  return err_flag;
}

// ----------------- I/O Functions

// Write out a matrix in default format
template<class eltype, class mat_type>
void write(std::ostream &out, const YA_BaseT &mat) {
  _ya_write_helper<YA_TRAIT::VME_MEMORY,
                   YA_TRAIT::VME_DIAGONAL>::write(out,mat.stv());
}

// Read in a matrix
template<class eltype, class mat_type>
int read(std::istream &in, YA_BaseT &mat) {
  int err_flag = 0;
  std::string token;
  in >> token;
  if (token=="VecMatSparse")
    err_flag=_ya_readsparse_helper<eltype,YA_TRAIT::VME_MEMORY>::read(in,mat.stv());
  else if (token=="VecMatDiag")
    err_flag=_ya_readdiag_helper<eltype,mat_type,YA_BaseT>(in,mat);
  else if (token=="VecMatSym")
    err_flag=_ya_readsym_helper<eltype,mat_type,YA_BaseT>(in,mat);
  else if (token=="VecMat")
    err_flag=_ya_read_helper<eltype,mat_type,YA_BaseT>(in,mat);
  else if (token=="YA_Binary")
    err_flag=_ya_binary_helper<YA_TRAIT::VME_MEMORY,
                              ya_direct_mem<eltype,eltype>::ans>::read(in,mat);
  else {
    ya_putback(in,token);
    YA_DynT tmat;
    err_flag=_ya_readascii_helper<eltype>(in,tmat);
    mat=tmat.T();
  }
  return err_flag;
}

// Const copy
template<class eltype, class mat_type>
int read(std::istream &in, const YA_WrapperT &mat) {
  int err_flag = 0;
  std::string token;
  in >> token;
  if (token=="VecMatSparse")
    err_flag=_ya_readsparse_helper<eltype,YA_TRAIT::VME_MEMORY>::read(in,mat.stv());
  else if (token=="VecMatDiag")
    err_flag=_ya_readdiag_helper<eltype,mat_type,const YA_WrapperT>(in,mat);
  else if (token=="VecMatSym")
    err_flag=_ya_readsym_helper<eltype,mat_type,const YA_WrapperT>(in,mat);
  else if (token=="VecMat")
    err_flag=_ya_read_helper<eltype,mat_type,const YA_WrapperT>(in,mat);
  else if (token=="YA_Binary")
    err_flag=_ya_binary_helper<YA_TRAIT::VME_MEMORY,
                              ya_direct_mem<eltype,eltype>::ans>::read(in,mat);
  else {
    ya_putback(in,token);
    YA_DynT tmat;
    err_flag=_ya_readascii_helper<eltype>(in,tmat);
    mat=tmat.T();
  }
  return err_flag;
}

// Write out a matrix
template<class eltype, class mat_type>
void write(std::ostream &out, const YA_BaseT &mat, const ya_sizet format) {
  if (format==YA_DEFAULT_IO)
    write(out,mat);
  else if (format==YA_BINARY_IO)
    _ya_binary_helper<YA_TRAIT::VME_MEMORY,
                      ya_direct_mem<eltype,eltype>::ans>::write(out,mat);
  else
    out << mat << std::endl;
}

// Read in a matrix
template<class eltype, class mat_type>
int read(std::istream &in, YA_BaseT &mat, const ya_sizet format) {
  if (format==YA_DEFAULT_IO)
    return read(in,mat);
  else if (format==YA_BINARY_IO)
    return _ya_binary_helper<YA_TRAIT::VME_MEMORY,
                             ya_direct_mem<eltype,eltype>::ans>::read(in,mat);
  else {
    int err_flag=0;
    YA_DynT tmat;
    err_flag=_ya_readascii_helper<eltype>(in,tmat);
    mat=tmat.T();
    return err_flag;
  }
}    

// Const copy
template<class eltype, class mat_type>
int read(std::istream &in, const YA_WrapperT &mat, const ya_sizet format) {
  if (format==YA_DEFAULT_IO)
    return read(in,mat);
  else if (format==YA_BINARY_IO)
    return _ya_binary_helper<YA_TRAIT::VME_MEMORY,
                             ya_direct_mem<eltype,eltype>::ans>::read(in,mat);
  else {
    int err_flag=0;
    YA_DynT tmat;
    err_flag=_ya_readascii_helper<eltype>(in,tmat);
    mat=tmat.T();
    return err_flag;
  }
}    

// Save a matrix to file in default format
template<class eltype, class mat_type>
int save(const std::string &filename, const YA_BaseT &mat) {
  std::ofstream out;
  int err_flag=ya_fileopen(out,filename);
  if (err_flag!=0)
    return err_flag;
  write(out,mat);
  err_flag=ya_fileclose(out);
  return err_flag;
}

// Load a matrix from file
template<class eltype, class mat_type>
int load(const std::string &filename, YA_BaseT &mat) {
  std::ifstream in;
  int err_flag=ya_fileopen(in,filename);
  if (err_flag!=0)
    return err_flag;
  err_flag=read(in,mat);
  if (err_flag!=0) {
    ya_fileclose(in);
    return err_flag;
  }
  err_flag=ya_fileclose(in);
  return err_flag;
}

// Const copy
template<class eltype, class mat_type>
int load(const std::string &filename, const YA_WrapperT &mat) {
  std::ifstream in;
  int err_flag=ya_fileopen(in,filename);
  if (err_flag!=0)
    return err_flag;
  err_flag=read(in,mat);
  if (err_flag!=0) {
    ya_fileclose(in);
    return err_flag;
  }
  err_flag=ya_fileclose(in);
  return err_flag;
}

// Save a matrix to a file
template<class eltype, class mat_type>
int save(const std::string &filename, const YA_BaseT &mat, const ya_sizet format) {
  std::ofstream out;
  int err_flag;
  if (format==YA_BINARY_IO)
    err_flag=ya_fileopenbinary(out,filename);
  else
    err_flag=ya_fileopen(out,filename);
  if (err_flag!=0)
    return err_flag;
  write(out,mat,format);
  err_flag=ya_fileclose(out);
  return err_flag;
}

// Load a matrix from file
template<class eltype, class mat_type>
int load(const std::string &filename, YA_BaseT &mat, const ya_sizet format) {
  std::ifstream in;
  int err_flag;
  if (format==YA_BINARY_IO)
    err_flag=ya_fileopenbinary(in,filename);
  else
    err_flag=ya_fileopen(in,filename);
  if (err_flag!=0)
    return err_flag;
  err_flag=read(in,mat,format);
  if (err_flag!=0) {
    ya_fileclose(in);
    return err_flag;
  }
  err_flag=ya_fileclose(in);
  return err_flag;
}

// Const copy
template<class eltype, class mat_type>
int load(const std::string &filename, const YA_WrapperT &mat, const ya_sizet format) {
  std::ifstream in;
  int err_flag;
  if (format==YA_BINARY_IO)
    err_flag=ya_fileopenbinary(in,filename);
  else
    err_flag=ya_fileopen(in,filename);
  if (err_flag!=0)
    return err_flag;
  err_flag=read(in,mat,format);
  if (err_flag!=0) {
    ya_fileclose(in);
    return err_flag;
  }
  err_flag=ya_fileclose(in);
  return err_flag;
}

}
#endif
