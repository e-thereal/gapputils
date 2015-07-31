SOURCES = deprecated/ImageReader.cpp \
          deprecated/ImageWriter.cpp \
          deprecated/MnistReader.cpp \
          OpenImage.cpp \
          OpenMnist.cpp \
          OpenNii.cpp \
          OpenNiiTensor.cpp \
          OpenTensor.cpp \
          SaveImage.cpp \
          SaveNii.cpp \
          SaveTensor.cpp \
          CopyNiftiHeader.cpp
		  
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
  
  QMAKE_POST_LINK += cp ${PWD}/${TARGET} ${GRAPEVINE_DEBUG_LIBRARY_PATH}/libgml.imaging.io.so
  
  message("Debug build.")
}

CONFIG(release, debug|release) {
  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
  
  QMAKE_POST_LINK += cp ${PWD}/${TARGET} ${GRAPEVINE_LIBRARY_PATH}/libgml.imaging.io.so
  
  message("Release build.")
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}
INCLUDEPATH += ${CUDA_INC_PATH}

LIBS += -Wl,-E -pg
LIBS += -L${RESPROG_LIB_PATH}
LIBS += -L${CUDA_LIB_PATH}

CONFIG(MIF) {

SOURCES += trace.cpp \
           SliceFromMif.cpp \
           OpenMif.cpp \
           SaveMif.cpp \
           deprecated/MifReader.cpp \
           deprecated/MifWriter.cpp

CONFIG(fornix, gpufarm|fornix|debian7) {
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/cmif
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/ctrace
	INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/carray
	LIBS += -lcmif_v5_3 -lutilities_v3_2
	
	message("Fornix build.")
}

CONFIG(gpufarm, gpufarm|fornix|debian7) {
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/cmif
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/utilities
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/ctrace
	INCLUDEPATH += /res1/software/x64/cmif_v5_3/carray
	LIBS += -lcmif_v5_3 -lutilities_v3_2
	
	message("Gpufarm build.")
}

CONFIG(debian7, gpufarm|fornix|debian7) {
  INCLUDEPATH += /res1/production/cmif_v5_4_debian7/cmif
  INCLUDEPATH += /res1/production/cmif_v5_4_debian7/utilities
  INCLUDEPATH += /res1/production/cmif_v5_4_debian7/ctrace
  INCLUDEPATH += /res1/production/cmif_v5_4_debian7/carray
  LIBS += -lcmif_v5_4 -lutilities_v3_3
  
  message("Debian 7 build.")
}

LIBS += -lcudart -lz

}

CONFIG(RAW) {
  SOURCES += OpenRaw.cpp
  LIBS += -lraw
}