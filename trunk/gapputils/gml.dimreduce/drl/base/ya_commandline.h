/***************************************************************************
                               ya_commandline.h
                             -------------------
                               W. Michael Brown

  Functions for CommandLine interface
 
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

#ifndef YA_COMMANDLINE_H
#define YA_COMMANDLINE_H

#include "commandline.h"

YALANAMESPACE

/// Add file types description to CommandLine manpage
/** Added under the title Matrix File Types **/
inline void ya_addftmanpage(CommandLine &cl) {
  const string form[39]={
    "Five file formats are available for matrix and vector I/O ",
    "depending on the type of matrix and format: full, symmetric, ",
    "sparse, diagonal, pretty, and binary format.\n\n",
    "The format for the full matrix is: \n\n",
    "\tVecMat \\fI<rows> <columns>\n",
    "\\fI\t<data at row1, col1>\n",
    "\\fI\t<data at row2, col1>\n",
    "\\fI\t<data at row3, col1>\n",
    "\\fI\t...\n",
    "\\fI\t<data at row2, col_n>\n",
    "\\fI\t<data at row1, col2>\n",
    "\\fI\t...\n\n",
    "The format for a symmetric matrix is: \n\n",
    "\tVecMatSym \\fI<rows>\n",
    "\\fI\t<data at row1, col1>\n",
    "\\fI\t<data at row2, col1>\n",
    "\\fI\t<data at row2, col2>\n",
    "\\fI\t<data at row3, col1>\n",
    "\\fI\t<data at row3, col2>\n",
    "\\fI\t<data at row3, col3>\n",
    "\\fI\t...\n\n",
    "for the lower triangle. The format for a diagonal matrix is: \n\n",
    "\tVecMatDiag \\fI<rows>\n",
    "\\fI\t<data at row1, col1>\n",
    "\\fI\t<data at row2, col2>\n",
    "\\fI\t<data at row3, col3>\n",
    "\\fI\t...\n\n",
    "for the diagonal. The format for a sparse matrix is: \n\n",
    "\tVecMatSparse \\fI<rows> <columns>\n",
    "\\fI\t<col_number*rows+row_number> <value>\n",
    "\\fI\t<col_number*rows+row_number> <value>\n",
    "\\fI\t...\n",
    "\tEND\n\n",
    "The format for a pretty file is:\n\n",
    "\\fI\t<data row1, col1> <data row1, col2> ... <data row1,col_n>\n",
    "\\fI\t<data row2, col1> <data row2, col2> ... <data row2,col_n>\n",
    "\\fI\t...\n",
    "Binary files can also be loaded and saved, but are not architecture ",
    "independent.\n\n"
  };
      
  cl.addtoman_chapter("MATRIX FILE FORMATS",39,form);
}

}
#endif
