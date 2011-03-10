SOURCES = main.cpp MainWindow.cpp Person.cpp PropertyReference.cpp ModelHarmonizer.cpp
HEADERS = MainWindow.h Person.h PropertyReference.h ModelHarmonizer.h

CONFIG += no_keywords debug
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += ../capputils
INCLUDEPATH += ../tinyxml
LIBS += -L/home/tombr/Projects/tinyxml/Debug -L/home/tombr/Projects/capputils/Debug -lcapputils -ltinyxml -lboost_signals