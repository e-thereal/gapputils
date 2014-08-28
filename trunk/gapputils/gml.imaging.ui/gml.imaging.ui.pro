SOURCES = ImageViewer.cpp \
          ImageViewerDialog.cpp \
          TensorViewer.cpp \
          TensorViewerDialog.cpp \
		  
HEADERS = ImageViewerDialog.h \
          TensorViewerDialog.h
          
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
  
  QMAKE_POST_LINK += cp ${PWD}/${TARGET} ${GRAPEVINE_DEBUG_LIBRARY_PATH}/libgml.imaging.ui.so
  
  message("Debug build.")
}

CONFIG(release, debug|release) {
  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
  
  QMAKE_POST_LINK += cp ${PWD}/${TARGET} ${GRAPEVINE_LIBRARY_PATH}/libgml.imaging.ui.so
  
  message("Release build.")
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}
INCLUDEPATH += ${CUDA_INC_PATH}
INCLUDEPATH += ${CUDASDK_INC_PATH}

LIBS += -Wl,-E -pg
LIBS += -L${RESPROG_LIB_PATH}
