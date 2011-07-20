SOURCES = FromRgb.cpp \
          Grid.cpp \
          GridDialog.cpp \
          GridLine.cpp \
          GridList.cpp \
          GridModel.cpp \
          GridPoint.cpp \
          GridPointItem.cpp \
          GridWidget.cpp \
          ImageWarp.cpp \
          ToRgb.cpp \
          AamBuilder.cpp \
          trace.cpp \
          SliceFromMif.cpp \
          ActiveAppearanceModel.cpp \
          AamTester.cpp \
          RectangleModel.cpp \
          Rectangle.cpp \
          RectangleDialog.cpp \
          RectangleWidget.cpp \
          RectangleItem.cpp \
          Cropper.cpp \
          AamFitter.cpp \
          AamGenerator.cpp \
          AamWriter.cpp \
          AamReader.cpp \
          AamMatchFunction.cpp
		  
HEADERS = FromRgb.h \
          Grid.h \
          GridDialog.h \
          GridLine.h \
          GridList.h \
          GridModel.h \
          GridPoint.h \
          GridPointItem.h \
          GridWidget.h \
          ImageWarp.h \
          ToRgb.h \
          RectangleDialog.h \
          RectangleWidget.h
          
TEMPLATE = lib          
CONFIG += no_keywords debug dll
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += /home/tombr/Projects
INCLUDEPATH += /home/tombr/include
INCLUDEPATH += /home/tombr/Programs/cuda/include
INCLUDEPATH += /home/tombr/Programs/cula/include
INCLUDEPATH += /home/tombr/Programs/NVIDIA_GPU_Computing_SDK/C/common/inc
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/cmif
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/ctrace
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/carray
LIBS += -Wl,-E
LIBS += -L/home/tombr/Projects/tinyxml/Debug
LIBS += -L"/home/tombr/Projects/capputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/gapputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/culib/Debug"
LIBS += -L"/home/tombr/Projects/optlib/Debug"
LIBS += -L"/home/tombr/Programs/cuda/lib"
LIBS += -L"/home/tombr/Programs/cula/lib"
LIBS += -L/home/tombr/lib
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem -loptlib -lculib -lcudart -lcublas
LIBS += -lcmif_v5_3 -lutilities_v3_2 -lz