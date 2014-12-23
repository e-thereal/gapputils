// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the ALGLIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ALGLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#ifdef ALGLIB_EXPORTS
//#define ALGLIB_API __declspec(dllexport)
#else
//#define ALGLIB_API __declspec(dllimport)
#pragma comment(lib, "alglib")
#endif
#else
//#define ALGLIB_API
#endif

#define ALGLIB_API

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
