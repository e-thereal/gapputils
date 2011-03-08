SOURCES = main.cpp MainWindow.cpp Person.cpp
HEADERS = MainWindow.h Person.h

QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += ../capputils
INCLUDEPATH += ../tinyxml
LIBS += -L/home/tombr/Projects/tinyxml/Debug -L/home/tombr/Projects/capputils/Debug -lcapputils -ltinyxml