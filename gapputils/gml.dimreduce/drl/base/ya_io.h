/***************************************************************************
                                   ya_io.h
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

#ifndef YA_IO_H
#define YA_IO_H

/*! \file */

YALANAMESPACE

// --------- cout operator overloading
template<class eltype, class mat_type>
std::ostream & operator<< (std::ostream &out, const YA_BaseT &t);

enum {
  YA_DEFAULT_IO, //< Default matrix input or output
  YA_PRETTY_IO,  //< Space delimented matrix
  YA_BINARY_IO   //< Binary matrix
};

/// Open a file for input. Generates error ID \b 1-15 on fail.
int ya_fileopen(std::ifstream &in, const char *filename);
/// Open a file for input. Generates error ID \b 1-15 on fail.
int ya_fileopen(std::ifstream &in, const std::string &filename);
/// Open a binary file for input. Generates error ID \b 1-15 on fail.
int ya_fileopenbinary(std::ifstream &in, const std::string &filename);

/// Open a file for output. Generates error ID \b 2-15 on fail.
int ya_fileopen(std::ofstream &out, const std::string &filename);
/// Open a file for output. Generates error ID \b 2-15 on fail.
int ya_fileopenbinary(std::ofstream &out, const std::string &filename);
/// Open a file for append. Generates error ID \b 2-15 on fail.
int ya_fileopenapp(std::ofstream &out, const std::string &filename);
/// Open a file for output. Generates error ID \b 2-15 on fail.
int ya_fileopen(std::ofstream &out, const char *filename);

/// Close an input file. Generates error ID \b 10-15 on fail.
int ya_fileclose(std::ifstream &in);
/// Close an output file. Generates error ID \b 11-15 on fail.
int ya_fileclose(std::ofstream &out);

/// Write out a matrix and specify the format
/** See main documentation for formats 
  * \sa save() */
template<class eltype, class mat_type>
void write(std::ostream &out, const YA_BaseT &mat, const ya_sizet format);

/// Read a matrix and specify the format
/** See main documentation for formats 
  * \note An alternative read function exists that does not require
  *       specification of the format, but determines it at runtime
  * \sa load() */
template<class eltype, class mat_type>
int read(std::istream &out, YA_BaseT &mat, const ya_sizet format);

// Read into a wrapper
template<class eltype, class mat_type>
int read(std::istream &out, const YA_WrapperT &mat, const ya_sizet format);

/// Write out a matrix in default format
/** See main documentation for formats 
  * \sa save() */
template<class eltype, class mat_type>
void write(std::ostream &out, const YA_BaseT &mat);

/// Read a matrix
/** \sa load() */
template<class eltype, class mat_type>
int read(std::istream &out, YA_BaseT &mat);

// Read into a wrapper
template<class eltype, class mat_type>
int read(std::istream &out, const YA_WrapperT &mat);

/// Save a matrix to file in default format
/** See main documentation for formats 
  * \sa write() **/
template<class eltype, class mat_type>
int save(const std::string &filename, const YA_BaseT &mat);

/// Load a matrix from a file
/** \sa read() **/
template<class eltype, class mat_type>
int load(const std::string &filename, YA_BaseT &mat);

// Load a matrix into a wrapper
template<class eltype, class mat_type>
int load(const std::string &filename, const YA_WrapperT &mat);

/// Save a matrix to a file using the specified format
/** See main documentation for formats 
  * \sa write() **/
template<class eltype, class mat_type>
int save(const std::string &filename, const YA_BaseT &mat, const ya_sizet format);

/// Load a matrix from a file using the specified format
/** See main documentation for formats 
  * \note An alternative load function exists that does not require
  *       specification of the format, but determines it at runtime
  * \sa read() **/
template<class eltype, class mat_type>
int load(const std::string &filename, YA_BaseT &mat, const ya_sizet format);

// Load a matrix into a wrapper
template<class eltype, class mat_type>
int load(const std::string &filename, const YA_WrapperT &mat, const ya_sizet format);

/// Save a matrix to file in default format
/** See main documentation for formats 
  * \sa write() **/
template<class eltype, class mat_type>
int save(const char *filename, const YA_BaseT &mat) {
  return save(std::string(filename),mat);
}

/// Save a matrix to a file using the specified format
/** See main documentation for formats 
  * \sa write() **/
template<class eltype, class mat_type>
int save(const char *filename, const YA_BaseT &mat, const ya_sizet format) {
  return save(std::string(filename),mat,format);
}

/// Load a matrix from a file
/** \sa read() **/
template<class eltype, class mat_type>
int load(const char *filename, YA_BaseT &mat) {
  return load(std::string(filename),mat);
}

// Load a matrix into a wrapper
template<class eltype, class mat_type>
int load(const char *filename, const YA_WrapperT &mat) {
  return load(std::string(filename),mat);
}

/// Load a matrix from a file using the specified format
/** See main documentation for formats 
  * \note An alternative load function exists that does not require
  *       specification of the format, but determines it at runtime
  * \sa read() **/
template<class eltype, class mat_type>
int load(const char *filename, YA_BaseT &mat, const ya_sizet format) {
  return load(std::string(filename),mat,format);
}

// Load a matrix into a wrapper
template<class eltype, class mat_type>
int load(const char *filename, const YA_WrapperT &mat, const ya_sizet format) {
  return load(std::string(filename),mat,format);
}

}
#endif
