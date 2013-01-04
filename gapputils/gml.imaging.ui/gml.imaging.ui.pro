SOURCES = ImageViewer.cpp \
          ImageViewerDialog.cpp
		  
HEADERS = ImageViewerDialog.h
          
TEMPLATE = lib
TARGET = gml.imaging.ui
CONFIG += no_keywords dll

CONFIG(debug, debug|release) {
  QMAKE_CXXFLAGS += -pg
  QMAKE_LFLAGS += -pg
  LIBS += -pg
  
  LIBS += -L"../../tinyxml/Debug"
  LIBS += -L"../../capputils/Debug"
  LIBS += -L"../../gapputils/Debug"
  LIBS += -lgapputilsd -lcapputilsd -ltinyxmld
  
  message("Debug build.")
}

CONFIG(release, debug|release) {
  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
  
  message("Release build.")
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}

LIBS += -Wl,-E -pg
LIBS += -L${RESPROG_LIB_PATH}
