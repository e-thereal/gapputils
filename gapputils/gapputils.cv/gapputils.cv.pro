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
          ImageRepeater.cpp \
          ImageAggregator.cpp \
          ImageViewer.cpp \
          ImageViewerDialog.cpp \
          Blurring.cpp \
          Transform.cpp \
          Transformation.cpp \
          Register.cpp \
          SimilarityMeasure.cpp \
          AggregatorFunction.cpp \
          SlidingWindowFilter.cpp \
          cuda_util.cpp \
          SplitSlices.cpp \
          CudaImageInterface.cpp \
          Convolve.cpp \
          Interfaces.cpp \
          QtImage.cpp
		  
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
          RectangleWidget.h \
          ImageViewerDialog.h
          
TEMPLATE = lib
TARGET = gapputils.cv
CONFIG += no_keywords dll

CONFIG(debug, debug|release) {
  QMAKE_CXXFLAGS += -pg
  QMAKE_LFLAGS += -pg
  LIBS += -pg
  
  LIBS += -L"../../tinyxml/Debug"
  LIBS += -L"../../capputils/Debug"
  LIBS += -L"../../gapputils/Debug"
  LIBS += -lgapputilsd -lcapputilsd -ltinyxmld
  
  LIBS += -L"../../culib/Debug"
  LIBS += -L"../../tbblas/Debug"
  LIBS += -L"../../optlib/Debug"
  LIBS += -L"../../regutil/Debug"
  LIBS += -L"../../gapputils.cv.cuda/Debug"
  LIBS += -lgapputils.cv.cuda -lregutil -loptlib -lculib -ltbblas
  
  message("Debug build.")
}

CONFIG(release, debug|release) {
  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
  
  LIBS += -L"../../culib/Debug"
  LIBS += -L"../../tbblas/Debug"
  LIBS += -L"../../optlib/Debug"
  LIBS += -L"../../regutil/Debug"
  LIBS += -L"../../gapputils.cv.cuda/Debug"
  LIBS += -lgapputils.cv.cuda -lregutil -loptlib -lculib -ltbblas
  message("Release build.")
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}
INCLUDEPATH += ${CUDA_INC_PATH}
INCLUDEPATH += ${CUDASDK_INC_PATH}
INCLUDEPATH += ${CULA_INC_PATH}

INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/cmif
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/ctrace
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/carray

LIBS += -Wl,-E -pg
LIBS += -L${CUDA_LIB_PATH}
LIBS += -L${CULA_LIB_PATH}
LIBS += -L${RESPROG_LIB_PATH}
LIBS +=  -lcudart -lcublas -lcufft -lcula_core -lcula_lapack
LIBS += -lcmif_v5_3 -lutilities_v3_2 -lz
