/***************************************************************************
                                m_constants.h
                             -------------------
                               W. Michael Brown

  Misc constants

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Aug 10 2005
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
//using namespace std;

YALANAMESPACE

#define MATHLIB_VER "0.2"

#ifndef PI
///\def PI
/// pi
#define PI 3.1415926535897932384626433832795
#endif
///\def TWOPI
/// pi*2
#define TWOPI 6.2831853071795862
///\def HALFPI
/// pi/2
#define HALFPI 1.5707963267948966
///\def DEGTORAD
/// Convert Degrees to Radians (pi/180)
#define DEGTORAD 0.017453292519943295
///\def SQRT_TWO
/// sqrt(2.0)
#define SQRT_TWO 1.4142135623730951
///\def SQRT_PI
/// sqrt(PI)
#define SQRT_PI 1.7724538509055159

#ifndef EPS
///\def EPS
/// Small number
#define EPS 1e-100
#endif

#ifndef INF
///\def INF
/// Large number
#define INF std::numeric_limits<double>::max()
#endif

/** \mainpage Math Library
  * \section intro Introduction
  * Math library with containers and operations for vectors, matrices, graphs,
  * cartesian coordinates, quaternions, Euler angles, support vector machine
  * models, grids, etc.
  *
  * \section vecmat Vectors and Matrices
  *
  * Use of the vector/matrix library requires the file vm.h to be included.
  * Additionally, within main a pointer to the static error class MUST
  * be defined:
  * \code
  * #include "vm.h"
  * #include "error.h"
  * Error error;
  * VMStatic::set_error(&error); \endcode
  *
  * There are two compile flags for working with vectors and matrices.
  * -DEBUG mode will include a variety of sanity checks in addition
  *  to checks for indexing out of bounds. It is highly suggested to develop
  *  code in DEBUG mode. DEBUG mode is specified by two optional compiler
  *  preprocessor directives: YA_DEBUG1 or YA_DEBUG2. YA_DEBUG2 gives 
  *  additional information on the error including the matrices involved,
  *  however might require mode code generation. It is recommended to use
  *  YA_DEBUG2.
  * -DYA_SAFE will force a temporary object to be created for operator= to prevent
  *  problems where the left hand side is not independent of the expression
  *  on the right hand side. This also incurs a run-time performance cost.
  *  For example, x=x+b; does not require a temporary to be created in order
  *  to obtain the correct answer. x=x*b might require a temporary to be
  *  created because the expression on the right can change elements in x
  *  required for the calculation of other elements in x. In SAFE mode, both
  *  expressions are OK. In order to achieve maximum performance, SAFE mode
  *  should not be used and the code should control the creation of temporaries
  *  (i.e. x=YA_MatD(x*b) );
  *
  * \subsection performance Performance
  *
  * The classes for vector/matrix operations make full use of static-virtual
  * functions and expression templates in order to allow for all functions to
  * be in-lined and to avoid the creation of temporaries. There is an important
  * issue to be considered with the use of expression templates. The expression
  * templates do not necessarily have any data associated with them. For
  * example, while a zeros matrix can be used with any of the operations
  * defined, a matrix of zeros is not physically stored in memory. At compile
  * time, any index into the matrix is replaced with a 0. Likewise, the
  * operation a+b does not create a matrix in memory with the results from
  * a+b, but rather replaces at compile time any index into a+b with the code
  * to add the individual elements of the matrices. This offers a large
  * performance boost because it avoids the unnecessary creation of temporaries
  * for every operation. However, sometimes temporaries are necessary to
  * improve efficiency. This will often be the case when multiple indexes into
  * the same element in a matrix are performed. Each time an element is
  * indexed, the compiler must calculate the result from the template.
  * In cases where the same index is used multiple times, it is often
  * beneficial to create the temporary. For example, x=a*b*c; is an inefficient
  * line of code. Instead, use:
  * \code temp=a*b;
  * x=temp*c; \endcode
  * or
  * \code x=YA_MatD(a*b)*c; \endcode
  * For assignment of matrices, memcpy is used wherever possible. This includes
  * assignment into and from column vector slices of matrices. For comparison
  * of matrices, memcmp is used wherever possible (including column vector
  * slices). For zeroing matrices, memset is used wherever possible (including
  * slices). In order for memcpy, etc. to be utilized for user defined types,
  * a specialized _vm_direct_mem_unsafe() must be defined (see vmstatic.h)
  *
  * By default, all matrices are column major.
  *
  * Routines for sparse vectors and matrices are not optimized. This is being
  * performed as needed.
  *
  * \subsection matrix_types Matrix Types
  *
  * YA_MatT can be used exclusively for all vectors and matrices for quick
  * coding. In order to generate more efficient code, a variety of matrix and
  * vector types are available:
  *
  * <ol>
  * <li> <b>Run-time Fixed-Size Matrix</b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  * \code
  * // 3x3 Matrix of doubles
  * YA_MatD matrix; matrix.setup(3,3);
  * YA_Mat<double> matrix; matrix.setup(3,3);
  * YA_MatD matrix(3,3);
  * 
  * // Double Column Vector of length 3
  * YA_MatD vector; vector.setup(3);
  * YA_MatD vector(3);
  * 
  * // Unsigned 3x3 matrix
  * YA_MatI int_matrix(3,3);
  * YA_Mat<int> int_matrix(3,3);
  * 
  * // Operator= will set the size of matrix as necessary
  * YA_MatD matrix=matrix2;
  * YA_MatD matrix(matrix2);
  * 
  * // etc.
  * \endcode
  *
  *
  * <li> <b>Run-time Fixed-Size Matrix (Number of Columns Known at Compile Time)</b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time/Compile-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  * \code
  * // Double Column Vector of Length 3
  * YA_VecD vector; vector.setup(3);
  * YA_ColD vector; vector.setup(3);
  * YA_Cols<double,1> vector; vector.setup(3);
  * 
  * // Unsigned Column Vector of Length 3
  * YA_VecI int_vector(3);
  * YA_ColI int_vector(3);
  * YA_Cols<int,1> int_vector(3);
  * 
  * // Double matrix with 4 columns
  * YA_Cols<double,4> matrix(10);
  * YA_Cols<double,4> matrix(10,4);
  * 
  * // etc.
  * \endcode
  *
  *
  * <li> <b>Run-time Fixed-Size Matrix (Number of Rows Known at Compile Time)</b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time/Compile-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  * \code
  * // Double Row Vector of Length 3
  * YA_RowD vector; vector.setup(3);
  * YA_Rows<double,1> vector; vector.setup(3);
  * 
  * // Unsigned Row Vector of Length 3
  * YA_RowI int_vector(3);
  * YA_Rows<int,1> int_vector(3);
  * 
  * // Double matrix with 4 rows
  * YA_Cols<double,4> matrix(10);
  * YA_Cols<double,4> matrix(4,10);
  * 
  * // etc.
  * \endcode
  *
  *
  * <li> <b>Compile-time Fixed-Size Matrix </b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Stack
  * - Size Specificaiton: \e Compile-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  * \code
  * // Double Row Vector of Length 3
  * YA_Fix<double,1,3> vector;
  * 
  * // Unsigned Row Vector of Length 3
  * YA_Fix<int,1,3> int_vector;
  * 
  * // Double 3x3 matrix
  * YA_Fix<double,3,3> matrix;
  * 
  * // etc.
  * \endcode
  *
  *
  * <li> <b>Dynamic-Sized Matrix</b>
  * - Storage: \e STL \e Vector
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time
  * - memcpy/memcmp/memset: \e No
  * - Element access/set: \e O(1)
  * \code
  * // 3x3 Matrix of doubles
  * YA_DynD matrix; matrix.setup(3,3);
  * YA_Dyn<double> matrix; matrix.setup(3,3);
  * YA_DynD matrix(3,3);
  * 
  * // Double Column Vector of length 3
  * YA_DynD vector; vector.setup(3);
  * YA_DynD vector(3);
  * 
  * // Unsigned 3x3 matrix
  * YA_DynI int_matrix(3,3);
  * YA_Dyn<int> int_matrix(3,3);
  * 
  * // Operator= will set the size of matrix as necessary
  * YA_DynD matrix=matrix2;
  * YA_DynD matrix(matrix2);
  * 
  * // Add an element to a vector
  * matrix.push_back(value);
  * 
  * // etc.
  * \endcode
  * 
  *
  *
  * <li> <b>Sparse Matrix</b>
  * - Storage: \e STL \e Map
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time
  * - memcpy/memcmp/memset: \e No
  * - Element access/set: \e O(log(n))
  * \code
  * // 3x3 Matrix of doubles
  * VM_SparseD matrix; matrix.setup(3,3);
  * VecMatSparse<double> matrix; matrix.setup(3,3);
  * VM_SparseD matrix(3,3);
  * 
  * // Double Column Vector of length 3
  * VM_SparseD vector; vector.setup(3);
  * VM_SparseD vector(3);
  * 
  * // Unsigned 3x3 matrix
  * VM_SparseU int_matrix(3,3);
  * VM_SparseU<int> int_matrix(3,3);
  * 
  * // Operator= will set the size of matrix as necessary
  * VM_SparseU matrix=matrix2;
  * VM_SparseU matrix(matrix2);
  *
  * // Sparse matrix of doubles that returns INF instead of zero
  * // -- note this has not been fully tested. some operations might
  * //    result in unexpected behavior.
  * VM_SparseINF infmatrix(3,3);
  *
  * // etc.
  *
  * // To set the default value of a sparse matrix to a value other than
  * // zero of INF, see the API.
  *
  * \endcode
  *
  *
  * \note If -DVM_FAST is defined during compilation, sparse matrices require 
  * temporary creation for operator= if the operands are not independent
  * \code
  * sparse_new=sparse_one+sparse_two; // OK
  *
  * sparse_new=sparse_new+sparse_two; // not OK
  * // use
  * sparse_new=VM_SparseD(sparse_new+sparse_two); // or create temporary
  * \endcode
  *
  *
  * 
  *
  *
  * <li> <b>Symmetric Matrix</b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap
  * - Size Specificaiton: \e Run-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  *
  * - The symmetric matrix only stores one triangle of the matrix
  * - 2D Access is slightly slower than for the full matrix, 1D is the same
  * - Care must be taken because repeated operations on elements on either
  *   triangle act on the same element in memory. (i.e. adding 2 to mat(3,1)
  *   and mat(1,3) results in both being +4). Because the number of elements
  *   stored in memory is not rows()*cols(), certain functions may exhibit
  *   unexpected behavior with symmetric matrices (as noted below). Due to the
  *   use of expression templates, it will be more efficient to set symmetric
  *   matrices equal to operations that return a symmetric matrix. For example
  *   VM_SymMatD x=a.T()*a; will require much fewer operations than
  *   YA_MatD x=a.T()*a; because only the elements on one triangle will
  *   ultimately be calculated.
  * \code
  * // 3x3 Matrix of doubles
  * VM_SymMatD matrix; matrix.setup(3,3);
  * VecMatSym<double> matrix; matrix.setup(3,3);
  * VM_SymMatD matrix(3,3);
  * 
  * // Unsigned 3x3 matrix
  * VM_SymMatU int_matrix(3,3);
  * 
  * // Loop over symmetric matrix elements (slightly slower)
  * for (int i=0; i<rows(); i++)
  *   for (int j=0; j<=i; j++)
  *     do_some_op(matrix(i,j),i,j);
  *     
  * // Loop over elements (slightly faster)
  * symi=0;
  * for (int i=0; i<rows(); i++)
  *   for (int j=0; j<=i; j++) {
  *     do_some_op(matrix(symi),i,j);
  *     symi++;
  *   }
  * 
  * // Loop over elements and treat diagonal differently
  * int symi=0;
  * for (int i=0; i<rows(); i++) {
  *   for (int j=0; j<i; j++) {
  *     do_some_op(matrix(symi),i,j);
  *     symi++;
  *   }
  *   symi++;
  * }
  * // Do the diagonal
  * symi=0;
  * for (int i=0; i<rows(); i++) {
  *   do_some_op(matrix(symi),i,i);
  *   symi+=i+2;
  * }
  * 
  * // etc.
  * \endcode
  *
  *
  * 
  * <li> <b>Symmetric Matrix with size known at Compile Time</b>
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Stack
  * - Size Specificaiton: \e Compile-time
  * - memcpy/memcmp/memset: \e Yes
  * - Element access/set: \e O(1)
  *
  * - See Symmetric Matrix above for details.
  * \code
  * // 3x3 Matrix of doubles
  * VecMatSymF<double,3> matrix;
  * 
  * // Unsigned 3x3 matrix
  * VecMatSymF<int,3> int_matrix(3,3);
  * \endcode
  *
  *
  * 
  * <li> <b>Scalar with matrix ops defined</b>
  * - Storage: \e Single \e value
  * - Memory: \e Stack
  * - Size Specificaiton: \e Compile-time
  * - memcpy/memcmp/memset: \e No
  * - Element access/set: \e O(1)
  * \code
  * // 1x1 Matrix of double 3.0
  * VM_ScalarD scalar(3.0);
  * VM_ScalarD scalar=3.0;
  * 
  * // 1x1 Matrix of int 3.0
  * VM_ScalarU scalar(3);
  * 
  * // 1x1 Matrix of arbitrary
  * VecMatScalar<int> scalar(3);
  *
  * // VM_Scalar can be used interchangeably with the underlying type:
  * double value=VM_ScalarD(3.0);
  * 
  * // etc.
  * \endcode
  *
  *
  * 
  * <li> <b>Constant Vectors and Matrices</b>
  * - Storage: \e None
  * - Memory: \e None
  * - Size Specificaiton: \e Run-time
  * - memcpy/memcmp/memset: \e No
  * - Element access/set: \e O(1)
  * \code
  * // Column vector of zeros (type double)
  * VM_ZerosV vector(3);
  * VM_ZerosV vector; vector.setup(3);
  * // Matrix of zeros (type double)
  * VM_Zeros matrix(3,3);
  * VM_Zeros matrix; matrix.setup(3,3);
  * // Column vector of ones (type double)
  * VM_OnesV vector(3);
  * VM_OnesV vector; vector.setup(3);
  * // Matrix of ones (type double)
  * VM_Ones matrix(3,3);
  * VM_Ones matrix; matrix.setup(3,3);
  * // Column vector of twos (type int)
  * VecMatConstV<int,2> vector(3);
  * // Matrix of twos (type int>
  * VecMatConst<int,2> matrix(3,3);
  * // Identity Matrix of doubles
  * VM_EyeD matrix(3,3);
  * // Identity Matrix of floats
  * VecMatEye<float> matrix(3,3);
  * 
  * // ------------ Counting Column Vector (Start known at compile time)
  * // Unsigned row vector counting from 0 to 2
  * VM_ZeroToN count(3);
  * // Unsigned row vector counting from 5 to 7
  * VMCountFromI<int,5> count(3);
  * // etc.
  * 
  * // ------------ Counting Column Vector (Start known at Run-time)
  * // Unsigned Row Vector that counts from zero to 2
  * VM_CountU count(3);
  * // Unsigned Row Vector that counts from 1 to 3
  * VM_CountU count(3,1);
  * VM_CountU count(1,":",3);
  * // For double
  * VM_CountD count(3);
  * // For arbitrary
  * VecMatCountV<int> count(3);
  * // etc.
  * 
  * // ------------ Counting Column Vector With Specified Increment
  * // Unsigned Row Vector that counts from 1 to 5 by 2
  * VM_SeriesU count(1,":",2,":",5);
  * // Integer series
  * VecMatSeriesV<int> count(5,":",-2,":",1);
  * // etc.
  * 
  * // ------------ Counting matrix
  * // Unsigned Row Vector that counts from zero to 2
  * VM_CountMatU count(3);
  * // Vector that counts from 1 to 3
  * VM_CountMatU count(3,1);
  * VM_CountMatU count(1,":",3);
  * // 3x3 Matrix that counts down columns from 1 to 9
  * VM_CountMatU count(3,3,1);
  * // For doubles...
  * VM_CountMatD count(3);
  * // For arbitrary
  * VecMatCount<int> count(3);
  * // etc.
  *
  * // ------------ Linearly spaced row vector
  * // Linearly spaced vector from 0 to 10 with length of 100 
  * VM_LinSpaceD lin(0,10);
  * // Linearly spaced vector from 0 to 10 with length of 200
  * VM_LinSpaceD lin(0,10,200);
  * // Same, but with floats
  * VMLinSpace<float> lin(0,10,200);
  *
  * \endcode
  *
  *
  *
  *</ol>
  *
  * \subsection typesum Summary of Matrix Types
  *
  * Types ended in D are double, types ending in U are int:
  *
  * - YA_MatD, YA_MatI = Matrix with Arbitrary Sized
  * - YA_RowD, YA_RowI = Row vector
  * - YA_ColD, YA_ColI, YA_VecD, YA_VecI = Column vector
  * - YA_DynD, YA_DynI = Matrix with dynamic size
  * - VM_SymMatD, VM_SymMatU = Symmetric Matrix
  * - VM_SparseD, VM_SparseU = Sparse Matrix
  * - VM_ScalarD, VM_ScalarU = Scalar Matrix
  *
  * - YA_Fix<number_type,rows,columns> = Compile-time fixed size matrix
  * - VecMatSymF<number_type,rows> = Compile-time fixed size symmetric matrix
  * - YA_Rows<number_type,rows> = Compile-time fixed number of rows
  * - YA_Cols<number_type,columns> = Compile-time fixed number of columns
  *
  * - VM_ZeroToN = Unsigned column vector that counts from 0 to n by 1
  * - VM_CountFromI<number_type,start_num> = vector that counts from start_num
  * - VM_CountD, VM_CountU = Vectors that count by 1 from some starting number
  * - VecMatCountV<number_type> = Vector that counts by 1 from some start
  * - VM_CountMatD, VM_CountMatU = Matrices that count by 1
  * - VecMatCount<number_type> = Matrix that counts by 1
  * - VM_SeriesD, VM_SeriesU = Matrix that counts by arbitrary increment
  * - VecMatSeriesV<number_type> = Matrix that counts by arbitrary increment
  * - VM_LinSpaceD = Linearly spaced vector of doubles
  * - VMLinSpace<number_type> = Linearly spaced vector of doubles
  *
  * - VM_ZerosV = Column vector of 0's
  * - VM_OnesV = Column vector of 1's
  * - VecMatConstV<number_type,number> = Column vector of number
  * - VM_Zeros = Matrix of 0's
  * - VM_Ones = Matrix of 1's
  * - VecMatConst<number_type,number> = Matrix of number
  * - VM_EyeD = Identity Matrix of double
  * - VecMatEye<float> = Identity Matrix of specific type
  *
  * \subsection constmatfun Functions that return constant matrices
  *
  * These are alternative constructors for constant matrices:
  *
  * \code
  * // Return a column vector of 5 0's (type double)
  * cout << vmzeros(5) << endl;
  *
  * // Return a column vector of 5 1's (type double)
  * cout << vmones(5) << endl;
  *
  * // Return identity matrix (type double)
  * cout << vmeye(5) << "or\n" << vmeye(5,5) << endl;
  *
  * // Return a row vector that counts from 0 to 4
  * cout << vmcount(5) << endl;
  *
  * // Return a row vector that counts from 1 to 5
  * cout << vmcount(1,":",5) << endl;
  *
  * // Return a row vector that counts from 1 to 5 by 2
  * cout << vmcount(1,":",2,":",5) << endl;
  *
  * // Return a linearly spaced row vector from 0.5 to 1.5 with length of 100
  * cout << vmlinspace(0.5,1.5) << endl;
  *
  * // Return a linearly spaced row vector from 0.5 to 1.5 with length of 200
  * cout << vmlinspace(0.5,1.5,200) << endl;
  *
  * \endcode
  *
  *
  *
  * \subsection casting Matrix Casting
  *
  * \code
  * // Static cast integer matrix to double values
  * YA_MatD double_mat=vm_cast<double>::sc(int_mat);
  *
  * // Cast a 1x1 matrix to a scalar
  * double dot_prod = scalar(vec_one.T()*vec_one);
  *
  * \endcode
  *
  *
  *
  * \subsection intiialization Matrix Initialization
  *
  * There are several methods for matrix initialization. For the most part,
  * they are general for all of the matrix types. Some limitations do apply,
  * however, for special matrix types (the number of rows must equal the
  * number of columns for symmetric matrices, etc.).
  * \code
  * // Empty 0x1 matrix
  * YA_MatD matrix;
  * 
  * // Arbitrary sized matrix (contents unspecified) ( 3 rows, 2 columns )
  * YA_MatD matrix(3,2);
  * 
  * // Column vector (contents unspecified) (length 9)
  * YA_MatD matrix(9);
  * 
  * // Set the size and contents equal to matrix2
  * YA_MatD matrix(matrix2);
  * YA_MatD matrix=matrix2;
  * 
  * // Clear the matrix and (optionally) the matrix size
  * matrix.clear()
  *
  * // Clear any old matrix and set up a column vector with 'length' elements
  * matrix.setup(const int length)
  *
  * // Clear any old matrix and set up a new one with specified rows/columns
  * matrix.setup(rows,cols);
  *
  * // Set the matrix to the identity matrix
  * matrix.eye();
  *
  * // Zero the entire matrix (using memset if possible)
  * // NOTE: YA_DynT (YA_Dyn) elements are initialized to zero during setup
  * matrix.zero_matrix();
  *
  * // Set all the elements in a matrix to a specified value
  * matrix.setall(value);
  *
  * // Set the matrix to random numbers >=0 and <=1
  * matrix.random();
  * 
  * // Set the matrix to random numbers >0 and <1
  * matrix.random_exc();
  *
  * // Set the matrix to a random normal distribution (mean=0, stddev=1)
  * matrix.random_normal();
  *
  * // Set the matrix to a random set of integers >=0 and <=7
  * matrix.random_int(7);
  *
  * // Set the size and contents of a matrix based on matlab notation:
  * // (All produce the same matrix)
  * matrix="[1 2 3; 4 6 8; 11 14 13]";
  * matrix="[ 1:3; 4:2:8; 11 14 13]";
  * matrix="[ 1 4 11; 2 6 14; 3 8 13]'";
  *
  * // Create a 1x1 matrix storing 2.0
  * matrix=2.0;
  * \endcode
  *
  * In general, matrices that have compile-time fixed sizes must have the
  * same size as the matrix being assigned. The exception is vectors, where
  * assignment to a fixed column vector from a row vector and vice-versa is
  * allowed.
  *
  * \subsection indexing Matrix indexing
  *
  * Indices and sizes for YALA++ are of type ya_sizet which is an int.
  *
  * 1D indexing can be performed with operator() or operator[]. 
  * The indexing is zero-based and in column major order. 
  * The indexing for a 3x3 matrix is:
  * \verbatim
  0 3 6
  1 4 7
  2 5 9
  \endverbatim
  * For a symmetric 3x3 matrix, the indexing is:
  * \verbatim
  0
  1 2
  3 4 5
  \endverbatim
  * The number of elements that can be accessed with 1D indexing is given
  * by the length(matrix) command or equivalently matrix.numel():
  * \code
  * for (int i=0; i<length(matrix); i++)
  *   if (matrix(i)<0)
  *     matrix(i)*=-1;
  * \endcode
  * 2D indexing can also be performed with operator().
  * \code
  * for (int i=0; i<matrix.cols(); i++)
  *   for (int j=0; j<matrix.rows(); j++)
  *     if (matrix(j,i)<0)
  *       matrix(j,i)*=-1;
  * \endcode
  * Index bounds checks are only performed when compiled with -DDEBUG. Methods
  * for forming a new matrix with multiple indices is given below (Matrix
  * Wrappers).
  *
  * \subsection iterators Matrix Iterators
  *
  * Matrix iterators allow for matrix indexing in a manner similar to
  * pointers. They can also be used on matrix wrappers (see below).
  *
  * \code
  * for (YA_MatD::iter=matrix.begin(); iter!=matrix.end(); iter++) {
  *   *iter+=5;
  *   cout << *iter << endl;
  * }
  * \endcode
  *
  * \subsection wrappers Matrix Wrappers
  *
  * Matrix wrappers do not store matrix data, but rather provide new ways for
  * indexing data in a "parent" matrix. For example, transpose(matrix) does
  * not copy any data, but indexes the data in the original matrix differently.
  * Therefore, operations on a matrix wrapper also alter the data in the
  * parent matrix. Additionally, the parent matrix must not go out of
  * scope before the matrix wrapper is "finished".
  *
  * Matrix wrappers include:
  * - Transpose
  * - Matrix Reshape
  * - Matrix Tiling
  * - Matrix Reindexing
  * - Matrix Slicing
  * - Diagonal matrix from a vector
  * - Matrix concatenation
  *
  * \code
  * // Transpose - matrix.T(), transpose(matrix)
  * // NOTE: 1D access is slightly slower for transpose wrapper
  * x=a.T()*a;
  * x=transpose(a)*a;
  * 
  * // Reshape a matrix (numel() of the matrix wrapper must be the same)
  * // NOTE: Behavior will be different on VecMatSym (VM_SymMat) types
  * YA_MatD a(3,3);
  * x=reshape(a,9,1)*reshape(a,1,9); // Row vector times column vector
  * 
  * // Tile a matrix
  * // NOTE: Matrix access will be slightly slower
  * YA_MatD x=repmat(a,3,2); // x has 3 times the rows in a, 2 times the cols
  * 
  * // Tile the rows of a matrix (repmat(reshape(a,1,a.numel()),3,1)
  * // NOTE: Faster than repmat
  * YA_MatD x=rowrep(a,3);
  * 
  * // Tile the columns of a matrix (repmat(reshape(a,a.numel(),1),1,3)
  * // NOTE: Faster than repmat
  * YA_MatD x=colrep(a,3);
  * 
  * // Reindex a matrix
  * // - The input matrix consists of 1D indices into the parent matrix
  * // - The wrapper matrix takes on a new size of the input matrix
  * YA_MatI newi="( 0 1 2; 3 4 5; 6 7 8)";
  * YA_MatD newmatrix=oldmatrix(newi); // newmatrix is transpose of old
  * 
  * YA_MatI newindices="(0:8)'";
  * YA_MatD newmatrix=oldmatrix(newi); // newmatrix is column vector of old
  * 
  * // Matrix slicing
  * matrix(":",1)+=10; // Add 10 to the second column of matrix
  * matrix(0,":")+=5;  // Add 5 to the first row of matrix
  * 
  * // Slice the first, third, and fifth columns (or rows) of matrix
  * YA_MatI slicei="(0 2 4)";
  * matrix(":",slicei)+=10;
  * matrix(slicei,":")+=5;
  * 
  * // Slice the upper left corner of a matrix;
  * YA_MatD matrix(8,8);
  * matrix(VM_ZeroToN(4),VM_ZeroToN(4))=5.0;
  *
  * // Diagonal matrix from a vector
  * // NOTE: Matrix access will be slightly slower
  * diag(matrix)=other_diagonal_matrix;
  * cout << diag(matrix) << endl;
  *
  * // Concatenate two matrices columnwise (number of rows is the same)
  * // NOTE: Matrix access will be slightly slower
  * // NOTE: You cannot concatenate a const matrix with a non-const matrix
  * // NOTE: You cannot concatenate a sparse matrix with a non-sparse matrix
  * // NOTE: You cannot concatenate a symmetric matrix with anything
  * cout << concat(matrix,matrix2) << endl;
  *
  * // Concatenate two matrices rowwise (number of columns is the same)
  * // NOTE: Matrix access will be slightly slower
  * // NOTE: You cannot concatenate a const matrix with a non-const matrix
  * // NOTE: You cannot concatenate a sparse matrix with a non-sparse matrix
  * cout << concat(matrix,";",matrix2) << endl;
  *
  * // In order to concatenate matrices with different return types (i.e. a 
  * // const and a non-const matrix or a sparse matrix, you can use temporaries
  * // or copy into the column slices of another matrix:
  * cout << concat(regular_matrix,YA_MatD(sparse_matrix)) << endl;
  *
  * \endcode
  *
  * All wrappers allow indexing or modifying data in the original parent matrix
  *
  * When a matrix wrapper is assigned to a scalar, each element in the wrapper
  * is assigned the value:
  * \code
  * // Assign row to be 3.0
  * matrix(2,":")=3.0;
  * \endcode
  *
  * \subsection Matrix I/O
  *
  * Space delimeted matrices can be output via the overloaded operator<<.
  * A newline is NOT placed at the end of the last line of a matrix to
  * allow for inlining of row vectors.
  * \code
  * cout << matrix << endl;
  * \endcode
  *
  * Matrices can be saved to disk in the same ascii format:
  * \code
  * matrix.writefile("outputname.ext",VM_PRETTY_IO);
  *   // or
  * save(matrix,"outputname.ext",VM_PRETTY_IO);
  * \endcode
  *
  * To write matrices in the default format:
  * \code
  * ofstream out;
  * out.open("outputname.ext");
  * matrix.write(out);
  * matrix2.write(out);
  * out.close();
  *   // or
  * matrix.writefile("outputname.ext");
  *   // or
  * matrix.writefile("outputname.ext",VM_DEFAULT_IO);
  *   // or
  * save(matrix,"outputname.ext");
  *   // or
  * save(matrix,"outputname.ext",VM_DEFAULT_IO);
  * \endcode
  *
  * To read a matrix from disk in any of these formats:
  * \code
  * ifstream in;
  * in.open("inputname.ext");
  * matrix.read(in);
  * matrix2.read(in);
  * in.close();
  *   // or
  * matrix.readfile("inputname.ext");
  *   // or
  * load(matrix,"inputname.ext");
  * \endcode
  *
  * Note that all I/O can be performed with the wrappers
  * (i.e. to read in the columns of a matrix from separate files)
  * \code
  * cout << matrix(":",2) << endl;
  *
  * matrix(":",2).readfile("column3.txt");
  *
  * // etc.
  * 
  * \endcode
  *
  * \subsection rand Random Number Generation
  *
  * The default random number generator is the Mersenne Twister with a
  * period of 2^19937-1. Alternative random number generators can be
  * specified using a compiler directive to define VMRNG. For 
  * native cstd random number generation use -DVMRNG=MathCStdRNG when
  * compiling. One seed is utilized for the entire library. The following
  * commands are available for random number generation:
  *
  * \code
  * // Seed the RNG with the time
  * vm_seed();
  *
  * // Seed the RNG with a number
  * vm_seed(41283243);
  *
  * // Fill a matrix with random numbers >=0 and <=1:
  * YA_MatD matrix(4,3);
  * matrix.random();
  * // --- or
  * YA_MatD matrix=vm_random(4,3);
  *
  * // Fill a matrix with random numbers >0 and <1:
  * YA_MatD matrix(4,3);
  * matrix.random_exc();
  * // --- or
  * YA_MatD matrix=vm_random_exc(4,3);
  *
  * // Fill a matrix with a normal distribution of random numbers
  * // by Box-Muller method
  * // - Mean is 0 and standard deviation is 1
  * YA_MatD matrix(4,3);
  * matrix.random_normal();
  * // --- or
  * YA_MatD matrix=vm_random_normal(4,3);
  *
  * // Fill a matrix with random int integers >=0 and <=max
  * YA_MatI matrix(4,3);
  * matrix.random_int(5);
  * // --- or
  * YA_MatI matrix=vm_random_int(4,3,5);
  * 
  * \endcode
  *
  * The following operations return an single int or double
  * random number:
  *
  * \code
  * // random integer
  * int randnum=vm_rand();
  * 
  * // random integer >=0 and <=max
  * int randnum=vm_rand(5);
  *
  * // random double >=0 and <=1
  * double randnum=vm_rand01();
  * 
  * // random double >0 and <1
  * double randnum=vm_rand01_exc();
  * 
  * // normal random with mean 0 and standard deviation 1
  * double randnum=vm_normal();
  *
  * \endcode
  *
  * \subsection cblas BLAS interface
  *
  * Level 1, 2 and 3 BLAS functions are available through a templated
  * wrapper of the GSL-1.9 CBLAS Library (This library is included in
  * unmodified form in the /cblas subdirectory.) The library can be
  * used with arbitrary matrix types by omitting the prefix:
  * \code
  * cout << cblas_dot(10,vmcount(10),1,vmcount(10),1);
  * \endcode
  *
  * Alternatively, prefixed cblas routines can be called for float and
  * double precision numbers if the underlying type is a c-style array:
  * \code
  * #include <cblas/gsl_cblas.h>
  *
  * YA_MatD one=vmcount(10);
  * YA_MatD two=vmcount(10);
  * cout << cblas_ddot(10,one.c_ptr(),1,two.c_ptr(),1);
  * \endcode
  * The latter is required if you wish to use optimized BLAS libraries
  * such as ATLAS.
  *
  *
  * \subsection unary Unary Elementwise Operations
  *
  * The following perform standard C unary operations elementwise on a matrix.
  * Almost all will return expression templates, so temporary creation
  * may be beneficial in some cases to improve efficiency.
  *
  * \verbatim
  -,++,--,!,~,abs,acos,acosh,asin,asinh,atan,atanh,ceil,cos,cosh,erf,erfc,
  exp,exp2,expm1,finite,floor,isinf,isnan,lgamma,log,log10,log1p,sin,
  sinh,sqrt,tan,tanh
  \endverbatim
  *
  * \code
  * // Examples of operations that do not need a temporary for efficiency
  * x=cos(a)+b;
  * x=cos(a)-b;
  * x=cos(a).dot(b);
  * x=cos(a).dot_star(b);
  * // Here, a temporary can prevent repeated cos calculations each time
  * // an element is indexed:
  * x=cos(a)*b;              // LESS efficient
  * x=YA_MatD(cos(a))*b; // MORE efficient
  * \endcode
  *
  * The following additional operations are available:
  *
  * \code
  * // ---- log2
  * x=log_2(x);
  *
  * // ---- Square each element in the matrix
  * x=square(x);
  *
  * // ---- Find (return 1D indices to non-zero elements in a matrix)
  * YA_MatI indices=find(matrix);
  * YA_MatI indices=find(matrix==6.0);
  * YA_MatI indices=find(matrix<6.0);
  * 
  * \endcode
  * 
  * \subsection unarymat Unary Matrix Operations
  *
  * The following perform unary operations on a matrix.
  * Some will return expression templates, so temporary creation
  * may be beneficial in some cases to improve efficiency. Functions
  * with MATLAB commands commented next to them exhibit the same behavior
  * (max, sum, and prod will return row vectors acting upon each column, etc.)
  *
  * \code
  * // ---- length
  * int l=length(matrix); // or l=matrix.numel();
  * 
  * // ---- size
  * YA_MatI size=size(matrix); // MATLAB: size(matrix)
  * int rows=size(matrix,0); // also matrix.rows()
  * int cols=size(matrix,1); // also matrix.cols()
  * 
  * // ---- total (add up each element in matrix)
  * double t=total(matrix);  // also matrix.total()
  *
  * // ---- product (product of each element in matrix)
  * double p=product(matrix);  // also matrix.product()
  *
  * // ---- norm
  * double n=norm(matrix);
  * 
  * // ---- trace
  * double t=trace(matrix);
  *  
  * // ---- Sum
  * YA_MatD ans=sum(matrix); // MATLAB: ans=sum(matrix);
  * 
  * // ---- Product
  * YA_MatD ans=prod(matrix); // MATLAB: ans=prod(matrix);
  * 
  * // ---- Cumulative Product
  * YA_MatD ans=cumprod(matrix); // MATLAB: ans=cumprod(matrix);
  * 
  * // ---- Max
  * YA_MatD ans=max(matrix); // MATLAB: ans=max(matrix);
  * 
  * // ---- Max for matrix with indices
  * YA_MatD answer;
  * YA_MatI indices;
  * maxi(matrix,answer,indices); // MATLAB: (answer,indices)=max(matrix);
  * 
  * // ---- Max for vector with index
  * double answer;
  * int indices;
  * max(matrix,answer,indices); // MATLAB: (answer,indices)=max(matrix);
  * 
  * // ---- Sort
  * // **** Does not work for VecMatSym (VM_SymMat) matrix types
  * YA_MatD sorted=sort(matrix); // or sort(matrix,sorted);
  * matrix.sort();                  // MATLAB: matrix=sort(matrix);
  * 
  * // ---- Sort and return indices
  * // **** Does not work for VecMatSym (VM_SymMat) matrix types
  * YA_MatD answer;
  * YA_MatI indices;
  * sorti(matrix,answer,indices); // MATLAB: (answer,indices)=sort(matrix);
  * 
  * \endcode
  *
  * \subsection binary Binary Elementwise Operations
  *
  * The following are standard C, element-wise operations:
  * \verbatim
  +,-,*(for scalar),/(for scalar),==,!=,<,>,<=,>=,atan2,&&,||,%,<<,>>,&,|,pow
  \endverbatim
  *
  * The element-wise operations can be used with
  * a matrix and a scalar, or with two identically sized matrices:
  * \code
  * YA_MatD ans=2.0+matrix1;
  * 
  * YA_MatD ans=matrix1+2.0;
  * 
  * YA_MatD ans=matrix1+matrix2;
  * \endcode
  * The * and / operators due not perform elementwise operations when used with
  * two matrices. For this, see dot_star and dot_div below. The boolean
  * operators return a matrix containing the results of elementwise comparisons
  * (1 is true and 0 is false). This is for compatibility with matlab functions.
  * For binary matrix boolean operations see below.
  *
  * The following additional binary operators are available:
  * - (Note that ^ is the same as pow(), and is not a bitwise operation as in C)
  *
  *\code
  * // Element-wise multiplication
  * ans=a.dot_star(b);  // or ans=dot_star(a,b);
  * 
  * // Element-wise division
  * ans=a.dot_div(b); // or ans=dot_div(a,b);
  * 
  * // xor
  * ans=x_or(a,b);
  *   
  * // bitwise xor
  * ans=b_xor(a,b);
  *   
  * // Power
  * ans=pow(a,b);
  * ans=a^b;
  * \endcode
  *
  * \subsection binops Binary Matrix Operations
  *
  * \code
  * // Matrix multiplication (size checks only performed with -DDEBUG)
  * ans=a*b;
  * 
  * // Test if two matrices are identical
  * // -- Does not test if two matrices are the same size!
  * if (identical(one,two))
  *   cout << "The matrices are exactly the same.\n";
  *   
  * // ---- Dot product
  * // Matrices can be used for input as long as the lengths match
  * // If YA_BLAS defined, will use cblas_dot for supported inputs
  * double d=dot(one,two); // also d=one.dot(two);
  * 
  * \endcode
  *
  * \subsection setops Set Operations
  *
  * \code
  * // Unique elements - MATLAB: ans=unique(matrix);
  * YA_DynI ans=unique(matrix); 
  *
  * // Unique elements with indices (returns template) 
  * // MATLAB: [ans,indices]=unique(matrix);
  * YA_DynI indices;
  * YA_MatD ans=unique(matrix,indices); 
  *
  * // Unique elements with indices and backwards indices (returns template)
  * // MATLAB: [ans,indices,backind]=unique(matrix);
  * YA_DynI indices;
  * YA_MatI backind;
  * YA_MatD ans=unique(matrix,indices,backind);
  *
  * // Set Difference - MATLAB: ans=setdiff(matrix,matrix2);
  * YA_DynI ans=setdiff(matrix,matrix2);
  *
  * // Set Difference with Indices (Returns template)
  * // MATLAB: [ans,ind]=setdiff(matrix,matrix2);
  * YA_DynI ind;
  * YA_MatD ans=setdiff(matrix,matrix2,ind);
  *
  * // Set Union - MATLAB: ans=union(matrix,matrix2);
  * YA_DynI ans=setunion(matrix,matrix2);
  *
  * // Set Union with Indices - MATLAB: [ans,ia,ib]=union(matrix,matrix2)
  * YA_DynI ia,ib;
  * YA_DynI ans=setunion(matrix,matrix2,ia,ib);
  *
  * // Set Intersection - MATLAB: ans=intersect(matrix,matrix2);
  * YA_DynI ans=intersect(matrix,matrix2);
  *
  * // Set Instersection with Indices
  * // MATLAB: [ans,ia,ib]=intersect(matrix,matrix2)
  * YA_DynI ia,ib;
  * YA_DynI ans=intersect(matrix,matrix2,ia,ib);
  *
  * // Set XOR - MATLAB: ans=setxor(matrix,matrix2);
  * YA_DynI ans=setxor(matrix,matrix2);
  *
  * // Set XOR with Indices - MATLAB: [ans,ia,ib]=setxor(matrix,matrix2)
  * YA_DynI ia,ib;
  * YA_DynI ans=setxor(matrix,matrix2,ia,ib);
  * \endcode
  *
  * \subsection decomp Matrix Decompositions
  *
  * \code
  * // Cholesky Decomposition
  * // It is assumed that the matrix is symmetric
  * chol(matrix,lower);  // MATLAB: lower=chol(matrix)
  *
  * // Solves A*x=b with Cholesky Decomposition
  * chol_solve(a,x,b);
  *
  * // LU Decompositions
  * // L is stored in strict lower triangular part of the input
  * // U is stored in the upper triangle of the input
  * // p is a column vector storing the matrix permutation
  * lu(matrix,p,signum);
  *
  * // Solves A*x=b with LU Decomposition and stores answer in x
  * lu(A,p,signum);
  * lu_solve(A,p,x);
  *
  * // Invert A using LU Decomposition
  * lu(A,p,signum);
  * lu_invert(A,p,ans);
  * // ---- or
  * inv(A,ans);
  * // ---- or
  * ans=inv(A);
  * \endcode
  *
  * \subsection eigens Eigen Decomposition
  *
  * Various routines are provided for eigen decomposition, however, linking to
  * a fortran LAPACK library is required to take advantage of the various
  * routines as intended. Additionally, YA_LAPACK should be defined during
  * compiling (-DYA_LAPACK) in order to actually use LAPACK for these routines.
  * If YA_LAPACK is not defined, much slower internal routines will be used
  * for eigen decomposition. These include a simple driver for eigen 
  * decomposition and eigen decomposition with the power method. Additionally,
  * if the matrices storing the eigenvectors and eigenvalues do not use
  * c-style arrays with a LAPACK supported number type, the internal routines
  * will be utilized. All routines assume a square symmetric matrix as input.
  *
  * \code
  * YA_MatD input;
  * input.load("test_matrix.txt");
  * YA_MatD vectors;
  * YA_ColD values;
  *
  * // Perform eigen decomposition using divide-and-conquer
  * // - Eigen values are returned in ascending order
  * eig(input,vectors,values); // or
  * eig(input,values);
  *
  * // Perform eigen decomposition using simple driver (slower, less memory)
  * // - Eigen values are returned in ascending order
  * eigev(input,vectors,values); // or
  * eigev(input,values);
  *
  * // Perform eigen decomposition using Relatively Robust Representations
  * // - Eigen values are returned in ascending order
  * // - Tolerance gives criteria for eigen value convergence into an 
  * //   interval [a,b] with width: tol + eps *   max( |a|,|b| )
  * // - Specifying 0 for tol uses the LAPACK default value
  * double tol=0;
  * eigrrr(input,vectors,values,tol);
  * eigrrr(input,values,tol);
  *
  * // Perform eigen decomposition on symmetric matrix using internal driver
  * // - Eigen values are returned in ascending order
  * VMSymMatD symmat(input);
  * input.eigens(vectors,values);
  *
  * // Perform eigen decomposition on symmetric matrix using internal driver
  * // - Eigen values are returned in descending order
  * VMSymMatD symmat(input);
  * input.eigens_hl(vectors,values);
  * 
  * // Find the largest 'dim' eigenvalues with Relatively Robust Representations
  * // - Eigen values are returned in ascending order
  * int dim=3;
  * double tol=0;
  * eigrrr(input,dim,vectors,values,tol); // or
  * eigrrr(input,dim,values,tol);
  *
  * // Find the largest 'dim' eigenvalues with power decomposition or evx driver
  * // - Eigen values are returned in descending order
  * // - maxiters is ignored if the LAPACK evx routine is used (-DYA_LAPACK)
  * int dim=3;
  * int maxiters=1000;
  * powereigens(input,dim,vectors,values,tol,max_iters); // or
  * powereigens(input,dim,values,tol,max_iters);
  * \endcode
  * 
  * If YA_LAPACK is defined, shortcuts to the direct LAPACK routines are also 
  * defined. These include syev(), syevd(), syevx(), and syevr().
  *
  *
  * The eigs() and eigs_small() commands can be used to find a subset of
  * eigenvalues and vectors using the EigOpts class. For eigs, the largest
  * 'dim' eigenvalues are returned in descending order. For eigs_small, the
  * smallest eigenvalues are returned in ascending order.
  *
  * \code
  * // Find the largest 'dim' eigenvalues with Relatively Robust Representations
  * eigs(input,dim,vectors,values); // or
  * eigs(input,dim,values);
  *
  * // Find the largest 'dim' eigenvalues with Relatively Robust Representations
  * EigenOptions opts; opts.method(VMEIGRRR); opts.tol(1e-8); opts.dim(3);
  * eigs(input,vectors,values,opts); // or
  * eigs(input,values,opts);
  *
  * // Find the largest 'dim' eigenvalues with Power Method or EVX driver
  * EigenOptions opts; opts.method(VMEIGPOWER); opts.tol(1e-8); opts.dim(3);
  * opts.max_iters(1000); // only for internal power method
  * eigs(input,vectors,values,opts); // or
  * eigs(input,values,opts);
  *
  * // Find the largest 'dim' eigenvalues with divide-and-conquer
  * EigenOptions opts; opts.method(VMEIGDQ); opts.dim(3);
  * eigs(input,vectors,values,opts); // or
  * eigs(input,values,opts);
  *
  * // Find the largest 'dim' eigenvalues with simple driver
  * EigenOptions opts; opts.method(VMEIGSIMPLE); opts.dim(3);
  * eigs(input,vectors,values,opts); // or
  * eigs(input,values,opts);
  *
  * // For each function listed, eigs_small() can be used to instead return
  * // the smallest dim eigen values.
  * eigs_small(input,vectors,values,opts);
  * \endcode
  *
  * \subsection members Other Matrix Member Functions
  *
  * \code
  * // Return a deep copy of this matrix (Can convert template to full matrix)
  * VM_SymMatD sym=matrix.T()*matrix;
  * YA_MatD sorted=sort(sym.copy());
  *
  * // Swap two elements based on 1D index
  * matrix.swap(1,4);
  *
  * // Swap two rows
  * matrix.swap_rows(1,3);
  *
  * // Random shuffle rows n times
  * matrix.random_rows(n);
  *
  * // Swap two columns
  * matrix.swap_cols(1,3);
  *
  * // Random shuffle columns n times
  * matrix.random_cols(n);
  *
  * // Transpose the matrix
  * matrix.transpose();
  *
  * // Reverse the columns in a matrix
  * matrix.reverse();
  *
  * // Reverse the the order of each element in a matrix
  * matrix.reverseall();
  *
  * // Column center the matrix
  * matrix.column_center();
  *
  * // Double center a matrix
  * matrix.double_center();
  *
  * // Return the indices that would result in a sorted matrix
  * // -- These are not 1D indices, but indices for each column
  * YA_MatI indices;
  * matrix.sort(indices);
  *
  * // Return the 1D indices that sort a matrix into a vector
  * // - The sorted vector is given by matrix(indices)
  * YA_VecI indices;
  * matrix.sortvector(indices);
  *
  * // Normalize a vector
  * vector.normalize();
  *
  * // Return the minimum element in a matrix
  * cout << "Minimum element is: " << matrix.minimum() << endl;
  *
  * // Return the maximum element in a matrix
  * cout << "Maximum element is: " << matrix.maximum() << endl;
  *
  * // Return the squared euclidean distance between two rows
  * cout << "Distance between row1 and row2 is: " << matrix.dist2(1,3) << endl;
  *
  * // Full Eigen decomposition
  * YA_VecD values;
  * YA_MatD vectors;
  * matrix.eigens(vectors,values)
  *
  * // Add an scalar to each element in the diagonal
  * matrix.pluseye(5.0);
  *
  * \endcode
  *
  *
  * \subsection stats Statistics Commands
  *
  * \code
  * // ---- Covarience
  * VM_SymMatD covmat=cov(one,two); // MATLAB: covmat=cov(one,two);
  * 
  * // ---- Correlation Coefficients
  * double corrc=corrcoef(one,two); // MATLAB: coorc=corrcoef(one,two);
  *
  * // ---- Compute the coefficient of determination
  * double rsq=rsquared(one,two);
  * 
  * // ---- Compute the root mean squared deviation (rows are points)
  * double ans=rmsd(one,two);
  *
  * // ---- Compute the distance residual (see API for dist_residual())
  * dist_residual(original_dists, new_crds, dims, corr_vec, rsquared);
  *
  * \endcode
  *
  * \subsection other Other Commands
  *
  * \code
  * // ---- Compute the squared distance matrix
  * // **** Rows in the input matrix are points, columns are dimensions
  * VM_SymMatD=dist2_mat(matrix);
  * 
  * // ---- Compute the distance matrix
  * // **** Rows in the input matrix are points, columns are dimensions
  * VM_SymMatD=dist_mat(matrix);
  * 
  * // ---- K-nearest neighbors between each point in matrix to all other points
  * kneighbors(matrix,k,output_neighbors,distances);
  *
  * // ---- K-nearest neighbors from each point in matrix to each point in nbors
  * kneighbors(matrix,nbors,k,output_neighbors,distances);
  *
  * // ---- Row vector of prime numbers less than or equal to 14
  * YA_MatI p=primes(14);
  *
  * // ---- Row vector with random permutation of 0-13
  * YA_RowI randslice=randperm(14); // MATLAB: randslice=randperm(14)-1;
  *   // or
  * YA_RowI randslice; randperm(14,randslice);
  *
  * \endcode
  *
  * \subsection plot Plotting
  *
  * Several plotting functions are available, mostly for debugging, via
  * GNUPlot. In order to comply with ANSI C++, plotting is performed via
  * a temporary file (482vm897.vmt) rather than a pipe. The file is 
  * deleted after plotting. The plotting commands are:
  *
  * \code
  * // ----------- Plotting with a dimensionless x-coordinate
  * // -- plot(matrix,title(s),style)
  * // -- where title(s) is a string with comma separated legend titles
  * // -- and style is a character: '-' for a line, 'o' for points, '.'
  * // -- for dots, and '+' for a line with points marking the data
  *
  * // Single plot of a row-vector or column vector
  * YA_MatD x=vmlinspace(0,10);
  * plot(x,"0-10",'-');
  *
  * // Multiple plots of each column in the input matrix 
  * YA_MatD y=vmlinspace(0,20);
  * plot(concat(x.T(),y.T()),"0-10,0-20",'-');
  *
  * // ----------- Plotting with a specified x-coordinate
  *
  * // Single plot of a column or row vector x-coordinate vs a column/row y
  * plot(x,cos(x),"cos",'-');
  *
  * // Multiple plots with a single column or row vector x-coordinates versus
  * // each column in the input y matrix
  * plot(x,concat( cos(x.T()) , sin(x.T()) ),"cos,sin",'-');
  *
  * // Multiple plots with the columns of the first matrix representing the
  * // x-coordinates and the columns of the second matrix representing the 
  * // corresponding y-coordinates.
  * plot(concat(x.T(),y.T()),concat( cos(x.T()) , sin(y.T()) ),"cos,sin20",'-');
  *
  * \endcode
  *
  * \subsection dimred Dimensionality Reduction
  *
  * Dimensionality reduction can currently be performed using 
  * principal component analysis (PCA), locally linear embedding (LLE),
  * multidimensional scaling (MDS) and Isomap. 
  * 
  * The input matrix consists of rows representing data
  * points (columns represent dimensions). All dimensionality reduction
  * can be performed using the dim_reduce() command based on options
  * stored in the VMRedData class.
  *
  * Also see eigenproject(), pcareduce(), llereduce(), mdsreduce(),
  * and isomap_reduce().
  *
  * \subsection preprocessor Preprocessor Directives
  *
  * -YA_DEBUG1 - Generates a slower code with various sanity checks and 
  *              checks for indices out of bounds
  * -YA_DEBUG2 - Same as YA_DEBUG1, but generates additional output regarding
  *              the matrices involved in an error. Recommended for development
  *
  *
  * \subsection commerrs Common Errors
  * <ol>
  * <li> 1. Problems with ostream and certain operators
  * Replace
  * \code
  * cout << (matrix1==matrix2) << endl;
  * \endcode
  * with
  * \code
  * cout << (matrix1==matrix2).copy() << endl;
  *
  * <li> 2. Comparison between distinct pointer types ... lacks a cast
  * Many functions require that the number type in matrices passed
  * be identical (YA_Mat<int> and YA_Mat<double> might not mix).
  * This can be fixed using vm_cast:
  * \code
  * // Generates error
  * plot(vm_count(10),double_matrix);
  * // OK
  * plot(vm_cast<double>::sc(vm_count(10)),double_matrix);
  * \endcode
  *
  * <\ol>
  * \subsection todo To Do
  *
  * - Standardize INF, NAN representations
  * - Efficiency for diagonal matrices
  * - Efficiency for sparse matrices
  * - Change allowance/efficiency for assignment between row/column vectors?
  * - SymMat eigens and eigens_hl in both unfixed and fixed
  * - fix 1D setup in fixed rows and columns
  * - add test and documentation for min
  * - Make sure fixed matrices and slices that end up fixed VME_USE_BLAS=0
  * - Switch isvector() to VME_VECTOR, same for diagonal, etc.
  * - Add documentation/test for copy_lower_triangle()
  * - Add documentation for cblas_op5, cblas_op3, and shortcuts for
  *   gemv, symv, ger, gemm, and symm
  * - Fix intel exp and log, dotinv; testcases and documentation
  * - Add float versions of cblas wrapper
  * - Add documentation for row_sum
  * - Check about VME_BLAS_TYPE, temporary creation in cblas_ops, and non-cblas types
  * - Documentation and test for col_rep and row_rep
  * - Fix forward and reverse map for AtE to allow fixed matrix output
  * - Add into documentation that cblas is used for *=,*, and VML for exp,log,dotinv
  * - Fix matrix multiplication with cblas/expressions
  * - Create a temporary creation function that does nothing for things that
  *   are already vecmat, etc.
  * - Remove calls to size() from loops
  * - Automatically call BLAS functions from copy_matrix
  * - Fix problem with isuppz on intel dsyver
  * - Change all/ to only include what's necessary
  * - Add templates to expressions for max, sum, etc. to improve efficiency
  *   for column vectors
  * - Add regression tests for diag* and *diag
  * - Add diag for matrix to return diagonal vector
  *   - Check memory functions for diag efficiency
  * - Use traits for the copy() return type
  * - Efficiency for slices (stride vectors, memcpy, etc)(multiple columns,rows)
  * - Add efficiency for nz loop template
  * - Move solve directory files into their own header.
  * - Add cblas_swap to swap routines for efficiency
  * - Add cblas_spmv to cblas_op5 for packed symmetric
  *
  * Inefficiency
  * - Transpose and linspace requires extra time for 1D index
  *
  * Notes:
  *  When to use template <class YA_BaseT> instead of template <class vm_type>
  *     1. To restrict a function in global namespace to YALA objects
  *        In this case, try to use YA_BaseT for a const argument. If YA_BaseT
  *        is used for a non-const argument that accepts wrappers, a duplicate
  *        const version may need to be created.
  *     3. The matrix is stored in a wrapper templated based on the matrix type
  *
  * cml_drl: todo
  *     Reference isomap and lle
  *     Allow for specification of input coordinates for MDS to allow future map
  **/


}

#endif
