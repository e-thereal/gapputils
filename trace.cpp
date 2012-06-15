#include "trace.hpp"

#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h> // for memset

char CTrace::s_256blanks[256];
bool CTrace::s_enabled = false;
bool CTrace::s_enableAll = false;
std::set<std::string> CTrace::s_enabledClasses;
std::vector<const char*> CTrace::s_scopes;
int CTrace::s_fd = 1; // stdout

CTrace::CTraceInit CTrace::s_instance;

CTrace::CTraceInit::CTraceInit()
{
    memset(CTrace::s_256blanks, ' ', 256);
    std::ifstream in("./trace.ini");
    while (in.good()) {
        char buffer[1024];
        in.getline(buffer, 1023);
        if (strlen(buffer)==0) {
            // ignore, empty line
        } else if (buffer[0] == '#') {
            // ignore, comment
        } else if (!strncmp(buffer, "disableInTrace", 14)) {
            // ignored for performance reasons
        } else if (!strncmp(buffer, "disableOutTrace", 15)) {
            // ignored for performance reasons
        } else if (!strncmp(buffer, "file=", 5)) {
            CTrace::redirectOutput(&buffer[5]);
        } else if (!strncmp(buffer, "enableDebugAll", 14)) {
            CTrace::s_enableAll = true;
            CTrace::s_enabled = true;
        } else if (!CTrace::s_enableAll) {
            // if all classes are enabled, there is no point in 
            // storing the enabled class names.
            CTrace::s_enabledClasses.insert(buffer);
            CTrace::s_enabled = true;
        }
    }
}
CTrace::CTraceInit::~CTraceInit()
{
    if (CTrace::s_fd > 2) {
        close(CTrace::s_fd);
    }
}

#ifndef FASTCTRACE
//static 
void CTrace::staticHelper(const std::string& direction,
        const std::string& classname, const std::string& funcname,
        const std::string& param1, const std::string& param2,
        const std::string& param3)
{
    if (s_enabled && (s_enableAll || s_enabledClasses.find(classname)
            != s_enabledClasses.end())) {
        indent();
        write(s_fd, direction.c_str(), direction.size());
        write(s_fd, classname.c_str(), classname.size());
        write(s_fd, "::", 2);
        write(s_fd, funcname.c_str(), funcname.size());
        write(s_fd, "(", 1);
        if (param1 != "") {
            write(s_fd, param1.c_str(), param1.size());
            if (param2 != "") {
                write(s_fd, ", ", 2);
                write(s_fd, param2.c_str(), param2.size());
                if (param3 != "") {
                    write(s_fd, ", ", 2);
                    write(s_fd, param3.c_str(), param3.size());
                }
            }
        }
        write(s_fd, ")\n", 2);
        fdatasync(s_fd);
    }
}
//static 
void CTrace::in(const std::string& classname, const std::string& funcname,
        const std::string& param1, const std::string& param2,
        const std::string& param3)
{
    staticHelper("-->", classname, funcname, param1, param2, param3);
    s_scopes.push_back("");
}
//static 
void CTrace::out(const std::string& classname, const std::string& funcname,
        const std::string& param1, const std::string& param2,
        const std::string& param3)
{
    s_scopes.pop_back();
    staticHelper("<--", classname, funcname, param1, param2, param3);
}
//static 
void CTrace::comment(const std::string& classname, const std::string& funcname,
        const std::string& param1, const std::string& param2,
        const std::string& param3)
{
    staticHelper("", classname, funcname, param1, param2, param3);
}
#endif
void CTrace::redirectOutput(const std::string& file)
{
    if (file == "" || file == "cout") {
        s_fd = 1; // stdout
    } else if (file == "cerr") {
        s_fd = 2; // stderr
    } else {
        s_fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
	fchmod(s_fd, 0644);
    }
}
void CTrace::indent()
{
    // print two spaces for each indent
    int count = 2 * s_scopes.size();
    // write indenting whitespace in big chuncks for speed
    while (count > 256) {
        write(s_fd, s_256blanks, 256);
        count -= 256;
    }
    write(s_fd, s_256blanks, count);
}
void CTrace::comment(const char* text)
{
    if (m_enabled) {
        indent();
        // repeat the current scope to make this function easy to use
        write(s_fd, s_scopes.back(), strlen(s_scopes.back()));
        write(s_fd, " [", 2);
        write(s_fd, text, strlen(text));
        write(s_fd, "]\n", 2);
        fdatasync(s_fd);
    }
}
#ifndef FASTCTRACE
CTrace::CTrace() :
    m_enabled(false)
{
}
#endif
CTrace::CTrace(const char* className, const char* funcName, const char* par1,
        const char* par2, const char* par3) :
    m_enabled(false)
{
    // make this function fast if s_enabled is false, i.e. when ./trace.ini
    // does not exist
    if (s_enabled && (s_enableAll || s_enabledClasses.find(className)
            != s_enabledClasses.end())) {
        // here debugging is enabled for the current scope, so speed is
        // not that important, dynamic memory allocation is still fast
        // compared to file I/O.
        char* scope = new char[512];
        // tests are ordered based on the assumption that most calls
        // will be with zero or one parameters, and relatively
        // few with two or three.
        if (!par1) {
            snprintf(scope, 511, "%s::%s()", className, funcName);
        } else if (!par2) {
            snprintf(scope, 511, "%s::%s(%s)", className, funcName, par1);
        } else if (!par3) {
            snprintf(scope, 511, "%s::%s(%s, %s)", className, funcName, par1,
                    par2);
        } else {
            snprintf(scope, 511, "%s::%s(%s, %s, %s)", className, funcName,
                    par1, par2, par3);
        }
        // indent before push_back to get the correct indentation
        indent();
        write(s_fd, "-->", 3);
        write(s_fd, scope, strlen(scope));
        write(s_fd, "\n", 1);
        fdatasync(s_fd);

        // store the scope string so it won't need to be recomputed
        s_scopes.push_back(scope);
        m_enabled = true;
    }
}
#ifndef FASTCTRACE
// This function is slow, so it is disabled if the above symbol is defined.
// Once the code compiles with the symbol defined, the code can be recompiled
// without the symbol, the resulting code will be just as fast.
CTrace::CTrace(const std::string& className, const std::string& funcName,
        const std::string& par1, const std::string& par2,
        const std::string& par3) :
    m_enabled(false)
{
    // make this function fast if s_enabled is false, i.e. when ./trace.ini
    // does not exist
    if (s_enabled && (s_enableAll || s_enabledClasses.find(className)
            != s_enabledClasses.end())) {
        // here debugging is enabled for the current scope, so speed is
        // not that important, dynamic memory allocation is still fast
        // compared to file I/O.
        char* scope = new char[512];
        // tests are ordered based on the assumption that most calls
        // will be with zero or one parameters, and relatively
        // few with two or three.
        if (!par1.size()) {
            std::string acc = className + "::" + funcName + "()";
            snprintf(scope, 511, acc.c_str());
        } else if (!par2.size()) {
            std::string acc = className + "::" + funcName + "(" + par1 + ")";
            snprintf(scope, 511, acc.c_str());
        } else if (!par3.size()) {
            std::string acc = className + "::" + funcName + "(" + par1 + ", "
                    + par2 + ")";
            snprintf(scope, 511, acc.c_str());
        } else {
            std::string acc = className + "::" + funcName + "(" + par1 + ", "
                    + par2 + ", " + par3 + ")";
            snprintf(scope, 511, acc.c_str());
        }
        // indent before push_back to get the correct indentation
        indent();
        write(s_fd, "-->", 3);
        write(s_fd, scope, strlen(scope));
        write(s_fd, "\n", 1);
        fdatasync(s_fd);

        // store the scope string so it won't need to be recomputed
        s_scopes.push_back(scope);
        m_enabled = true;
    }
}
#endif

CTrace::~CTrace()
{
    if (m_enabled) {
        const char* scope = s_scopes.back();
        s_scopes.pop_back();

        // indent after pop_back to get the correct indentation
        indent();
        write(s_fd, "<--", 3);
        write(s_fd, scope, strlen(scope));
        write(s_fd, "\n", 1);
        fdatasync(s_fd);
        delete scope;
    }
}
