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
          AamMatchFunction.cpp \
          AamBuilder2.cpp \
          AamEcdnll.cpp \
          ImageCombiner.cpp \
          AamTester2.cpp \
          AamUtils.cpp \
          AamAnalyser.cpp \
          GridImagePair.cpp \
          Vector.cpp \
          ImageSaver.cpp \
          FromHsv.cpp \
          ToHsv.cpp \
          AamCreator.cpp \
          Resample.cpp \
          AamResample.cpp \
          SliceToFeatures.cpp \
          FeaturesToMif.cpp \
          FeaturesToImage.cpp \
          ImageToMif.cpp \
          StackImages.cpp \
          Checkerboard.cpp \
          ImageRepeater.cpp
		  
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
TARGET = gapputils.cv
CONFIG += no_keywords debug dll
QMAKE_CXXFLAGS += -std=c++0x -pg
QMAKE_LFLAGS += -pg
INCLUDEPATH += /home/tombr/Projects
INCLUDEPATH += /home/tombr/include
INCLUDEPATH += /res1/software/cuda/include
INCLUDEPATH += /res1/software/cula/include
INCLUDEPATH += /res1/software/NVIDIA_GPU_Computing_SDK/C/common/inc
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/cmif
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/ctrace
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/carray
LIBS += -Wl,-E -pg
LIBS += -L/home/tombr/Projects/tinyxml/Debug
LIBS += -L"/home/tombr/Projects/capputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/gapputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/culib/Debug"
LIBS += -L"/home/tombr/Projects/optlib/Debug"
LIBS += -L"/home/tombr/Projects/gapputils.cv.cuda/Debug"
LIBS += -L"/res1/software/cuda/lib"
LIBS += -L"/res1/software/cula/lib"
LIBS += -L/home/tombr/lib
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem -lgapputils.cv.cuda -loptlib -lculib -lcudart -lcublas -lcufft -lcula_core -lcula_lapack
LIBS += -lcmif_v5_3 -lutilities_v3_2 -lz