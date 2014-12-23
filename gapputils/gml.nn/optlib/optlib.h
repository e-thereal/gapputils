// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the OPTLIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// OPTLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.

#ifndef _OPTLIB_OPTLIB_H_
#define _OPTLIB_OPTLIB_H_

#ifdef _WIN32
#ifdef OPTLIB_EXPORTS
//#define OPTLIB_API __declspec(dllexport)
#else
//#define OPTLIB_API __declspec(dllimport)
#pragma comment(lib, "optlib")
#endif
#else
//#define OPTLIB_API
#endif
#define OPTLIB_API

const double Gold = 0.381966011; ///< golden = (3 - sqrt (5)) / 2
const double ExpGold = 1.618034; ///< 2 / (sqrt (5) - 1) : for Brent's exponential search

#endif
