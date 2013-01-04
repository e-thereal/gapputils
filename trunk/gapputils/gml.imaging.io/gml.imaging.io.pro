SOURCES = ImageReader.cpp
		  
HEADERS = 
          
TEMPLATE = lib
TARGET = gml.imaging.io
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

CONFIG(MIF) {

SOURCES += trace.cpp \
           SliceFromMif.cpp \
           MifReader.cpp

CONFIG(fornix, gpufarm|fornix) {
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/cmif
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/ctrace
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/carray
	message("Fornix build.")
}

CONFIG(gpufarm, gpufarm|fornix) {
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/cmif
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/utilities
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/ctrace
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/carray
	message("Gpufarm build.")
}

LIBS += -lcmif_v5_3 -lutilities_v3_2 -lz

}
