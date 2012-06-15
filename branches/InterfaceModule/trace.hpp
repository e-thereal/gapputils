#ifndef CTRACE_HPP_
#define CTRACE_HPP_

#include <string>
#include <set>
#include <vector>

/**
 * This is a reimplementation of the CTrace system for debugging support.
 * The old API consisted of two classes, one of which had a number of
 * static functions, and the other with non-static functions. In this
 * implementation, all functions are implemented by one class, and a 
 * typedef is used to alias both original class names to the new class.
 * 
 * The class uses static memory to manage most of its activity, the only
 * non-static memory is a bool in each instance, which is true if output
 * is enabled for the given class name.
 * 
 * A singleton helper class is used to read and parse the trace.ini 
 * initialization file. It is implemented as an inner class to the CTrace
 * class.
 * 
 * The new implimentation is optimised for speed in general, and for speed 
 * when debugging is disabled in particular. This has been accomplished 
 * using several techniques: Using low-level IO function calls open() and 
 * write(). Little error checking, since failures in debuggin 
 * code are not considered fatal. Arguments to the CTrace constructor
 * are char* rather than string, which is significantly faster.
 * 
 * The CTrace constructor that accepts STL strings is much slower than the
 * one accepting char*, mostly because of the amount of conversions that
 * are needed between strings and char*s. If the symbol FASTCTRACE is 
 * defined, the STL string version is disabled. This may mean that client
 * code must be modified to use CTrace (usually by calling c_str() on STL
 * strings before they are passed to CTrace), but the resulting code will
 * be much faster: A 75% reduction in run time of a CMIF5 test suite program
 * was measured, the only difference being which CTrace constructor was used.
 * 
 * Once the code has been modified so that it compiles cleanly with 
 * FASTCTRACE, it may be recompiled without this symbol defined, the 
 * resulting binary will be just as fast. 
 */
class CTrace
{
private:
    /**
     * This is a singleton helper class to CTrace. It handles reading 
     * ./trace.ini and initializes all static data in CTrace.
     */
    class CTraceInit
    {
    public:
        CTraceInit();
        ~CTraceInit();
    };
    static CTraceInit s_instance;
    /**
     * A buffer of 256 space characters used for generating the indents.
     */
    static char s_256blanks[256];
    /**
     * True if the file ./trace.ini exists. No output is produced if this
     * is not the case.
     */
    static bool s_enabled;
    /**
     * True if the file ./trace.ini contains the entry enableDebugAll.
     */
    static bool s_enableAll;
    /**
     * Set of all classes mentioned in ./trace.ini.
     */
    static std::set<std::string> s_enabledClasses;
    /**
     * Vector of all the scopes (class/function names) currently on the
     * call stack. Entries are pushed onto the vector by the CTrace
     * constructor and popped off by the destructor.
     */
    static std::vector<const char*> s_scopes;
    /**
     * The file handle for the output stream used for debugging output.
     */
    static int s_fd;
    /**
     * True if the current level on the call stack should generate output.
     * This is the only non-static data in a CTrace instance.
     */
    bool m_enabled;
    /**
     * Helper function for generating the set number of indents on the
     * current output stream.
     */
    static void indent();
    /**
     * Helper function for setting up the output file handle.
     */
    static void redirectOutput(const std::string& format);
#ifndef FASTCTRACE
    /**
     * Helper function for the static functions in(), out() and comment().
     */
    static void staticHelper(const std::string& direction,
            const std::string& classname, const std::string& funcname,
            const std::string& param1, const std::string& param2,
            const std::string& param3);
#endif

public:

#ifndef FASTCTRACE
    /**
     * Static function for generating a message indicating that a new
     * function has been entered. Do not use this function, allocate
     * an instance of CTrace instead.
     */
    static void in(const std::string& classname, const std::string& funcname,
            const std::string& param1="", const std::string& param2="",
            const std::string& param3="");
    /**
     * Static function for generating a message indicating that we are 
     * leaving a function. Do not use this function, allocate an instance of 
     * CTrace, and the CTrace destructor will handle this automatically.
     */
    static void out(const std::string& classname, const std::string& funcname,
            const std::string& param1="", const std::string& param2="",
            const std::string& param3="");
    /**
     * Static function for generating a comment for a given scope. Do
     * not use this function, use the non-static comment function below.
     */
    static void comment(const std::string& classname,
            const std::string& funcname, const std::string& param1="",
            const std::string& param2="", const std::string& param3="");
    /**
     * Default constructor. This will never generate any output, don't
     * use it.
     */
    CTrace();
#endif
    /**
     * Constructor generates a message indicating that we have entered a
     * given function. Use this!
     */
    CTrace(const char* className, const char* funcName,
            const char* par1 = NULL, const char* par2 = NULL,
            const char* par3 = NULL);
#ifndef FASTCTRACE
    /**
     * Constructor generates a message indicating that we have entered a
     * given function. This is significantly slower than the char* version 
     * above, use that when at all possible.
     * 
     * In one relatively realistic experiment (a CMIF test suite program),
     * the time penalty for using this function rather than the one
     * above was measured to be 410%, i.e. the "fast" char* version ran
     * in 47 seconds and the std::string version ran in 3 minutes 13 
     * seconds. This time difference was measured with an empty trace.ini 
     * file present.
     */
    CTrace(const std::string& className, const std::string& funcName,
            const std::string& par1 = "", const std::string& par2 = "",
            const std::string& par3 = "");
#endif
    /**
     * Destructor.
     */
    ~CTrace();
    /**
     * Generate the given comment on the output stream. The comment is
     * prepended with the correct number of indents and the name of the
     * current scope (class and function names, and parameter names).
     * Use this!
     */
    void comment(const char* text);
};

/**
 * Make sure both the old class names are aliases of the same class in the
 * new implementation.
 */
typedef CTrace CTraceIn;

#endif /*CTRACE_HPP_*/
