/***************************************************************************
                                  ya_misc.h
                             -------------------
                               W. Michael Brown

  Miscellaneous helper functions

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Sep 19 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_MISC_H
#define YA_MISC_H

#include <sstream>

YALANAMESPACE

/// Templated function for converting a given type to a string
/** Requires that operator << be overloaded for the type. **/
template <class t_type>
inline std::string ya_to_ascii(const t_type &value) {
   std::ostringstream out;
   out << value;
   return out.str();
}

// True if a character is whitespace
bool ya_whitespace(char c) {
  if (c==' ' || c=='\n' || c=='\t' || c=='\f' || c=='\r' || c=='\v')
    return true;
  return false;
}

// Check if a string is only whitespace
bool ya_whitespace(const std::string &s) {
  int sL=s.length();
  for (int i=0; i<sL; i++)
    if (!ya_whitespace(s[i]))
      return false;
  return true;
}

// Parse a white space separated string into tokens
void ya_get_tokens(const std::string &sline, std::vector<std::string> &tokens) {
  std::string token="";
  unsigned i=0;
  while (i<sline.length()) {
    if (ya_whitespace(sline[i])) {
      i++;
      continue;
    }
    while (i<sline.length()) {
      if (ya_whitespace(sline[i])) {
        tokens.push_back(token);
        token="";
        break;
      }
      token+=sline[i];
      i++;
    }
  }
  if (token!="")
    tokens.push_back(token);
}

// Parse a string into tokens
void ya_get_tokens(char delimiter, const std::string &sline,std::vector<std::string> &tokens) {
  std::string token="";
  unsigned i=0;
  while (i<sline.length()) {
    if (delimiter==sline[i]) {
      tokens.push_back(token);
      token="";
    } else
      token+=sline[i];
    i++;
  }
  if (token!="")
    tokens.push_back(token);
}

// Format a string to fit within a specified column width
void ya_format_fit(unsigned column_width, const std::string &input,
    std::vector<std::string> &output) {
  std::vector<std::string> forced;
  ya_get_tokens('\n',input,forced);
  for (unsigned i=0; i<forced.size(); i++) {
    std::string current_line;
    std::vector<std::string> tokens;
    ya_get_tokens(forced[i],tokens);
    for (unsigned j=0; j<tokens.size(); j++) {
      if (current_line.length()+tokens[j].length()<column_width) {
        current_line+=tokens[j]+' ';
      } else {
        if (tokens[j].length()+1>column_width) {
          unsigned this_count=column_width-current_line.length();
          current_line+=tokens[j].substr(0,this_count);
          output.push_back(current_line);
          current_line="";
          tokens[j]=tokens[j].substr(this_count,tokens[j].length()-this_count);
          j--;
        } else {
          output.push_back(current_line);
          current_line=tokens[j]+' ';
        }
      }
    }
    output.push_back(current_line);
  }
  return;
}

// Put a string back into an istream
void ya_putback(std::istream &in, const std::string &s) {
  if (s.length()==0)
    return;
  unsigned i=unsigned(s.length())-1;
  while (true) {
    in.putback(s[i]);
    if (i==0)
      return;
    i--;
  }
}

}
#endif
