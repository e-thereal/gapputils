#ifndef _REGUTIL_H_
#define _REGUTIL_H_

#ifdef _MSC_VER
#ifdef REGUTIL_EXPORTS
//#define CULIB_API __declspec(dllexport)
#else
//#define CULIB_API __declspec(dllimport)
#pragma comment(lib, "regutil")
#endif
#endif

#define REGUTIL_API

#endif
