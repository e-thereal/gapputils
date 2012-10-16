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
          Transform.cpp \
          Transformation.cpp \
          SimilarityMeasure.cpp \
          AggregatorFunction.cpp \
          SlidingWindowFilter.cpp \
          cuda_util.cpp \
          SplitSlices.cpp \
          CudaImageInterface.cpp \
          Convolve.cpp
		  
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
CONFIG += no_keywords release dll
QMAKE_CXXFLAGS += -std=c++0x -pg
QMAKE_LFLAGS += -pg
INCLUDEPATH += /res1/software/usr/include
INCLUDEPATH += /res1/software/x64/cuda/include
INCLUDEPATH += /res1/software/x64/cula/include
INCLUDEPATH += /res1/software/x64/NVIDIA_GPU_COMPUTING_SDK/C/common/inc
INCLUDEPATH += /res1/software/x64/cmif_v5_3/cmif
INCLUDEPATH += /res1/software/x64/cmif_v5_3/utilities
INCLUDEPATH += /res1/software/x64/cmif_v5_3/ctrace
INCLUDEPATH += /res1/software/x64/cmif_v5_3/carray
LIBS += -Wl,-E -pg
LIBS += -L"/res1/software/usr/lib64"
LIBS += -L"/home/tombr/Projects/gapputils.cv.cuda/Debian6.0.2"
LIBS += -L"/res1/software/x64/cuda/lib64"
LIBS += -L"/res1/software/x64/cula/lib64"
LIBS += -L"/res1/software/x64/cmif_v5_3/utilities"
LIBS += -L"/res1/software/x64/cmif_v5_3/cmif"
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem -lgapputilscvcuda -lculib -loptlib -lcudart -lcublas -lcufft -lcula_core -lcula_lapack -lcuda -ltbblas
LIBS += -lcmif_v5_3 -lutilities_v3_2 -lz
