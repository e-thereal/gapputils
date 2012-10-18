SOURCES = main.cpp \
		  MainWindow.cpp \
		  Person.cpp \
		  PropertyReference.cpp \
		  ModelHarmonizer.cpp \
		  PropertyGridDelegate.cpp \
		  Workbench.cpp \
		  ToolItem.cpp \
		  CableItem.cpp \
		  FilenameEdit.cpp \
		  ImageLoader.cpp \
		  ImageViewer.cpp \
		  ShowImageDialog.cpp \
		  DataModel.cpp \
		  Node.cpp \
		  Edge.cpp \
		  Workflow.cpp \
		  Controller.cpp \
		  DefaultInterface.cpp \
		  GenericViewer.cpp \
		  WorkflowItem.cpp \
		  MakeGlobalDialog.cpp \
		  GlobalProperty.cpp \
		  PopUpList.cpp \
		  GlobalEdge.cpp \
		  Expression.cpp \
		  trace.cpp \
		  HostInterface.cpp \
		  linreg.cpp \
		  Filename.cpp \
		  StringInterface.cpp \
		  ChecksumUpdater.cpp \
		  WorkflowController.cpp \
		  WorkflowUpdater.cpp \
		  Filenames.cpp \
		  NodeCache.cpp \
		  WorkflowToolBox.cpp \
		  PropertyGrid.cpp \
		  LogbookModel.cpp \
		  LogbookWidget.cpp \
		  GlobalPropertiesView.cpp \
		  ImageInterface.cpp \
		  ImagesInterface.cpp \
		  WorkbenchWindow.cpp
		  
HEADERS = MainWindow.h \
          Person.h \
          PropertyReference.h \
          ModelHarmonizer.h \
          PropertyGridDelegate.h \
          Workbench.h \
          FilenameEdit.h \
          ShowImageDialog.h \
          Workflow.h \
          WorkflowItem.h \
          GenericViewer.h \
          MakeGlobalDialog.h \
          PopUpList.h \
          ToolItem.h \
          WorkflowUpdater.h \
          WorkflowToolBox.h \
          PropertyGrid.h \
          LogbookModel.h \
          LogbookWidget.h \
          GlobalPropertiesView.h \
          WorkbenchWindow.h
          
RESOURCES = res.qrc
          
CONFIG += no_keywords console

debug {
  CONFIG -= release
  QMAKE_CXXFLAGS += -pg
  QMAKE_LFLAGS += -pg
  LIBS += -pg
  LIBS += -L"../../tinyxml/Debug"
  LIBS += -L"../../capputils/Debug"
  LIBS += -L"../../gapputils/Debug"
  LIBS += -lgapputilsd -lcapputilsd -ltinyxmld
}

release {
  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}
INCLUDEPATH += ${CUDA_INC_PATH}
INCLUDEPATH += ${CULA_INC_PATH}
DEFINES += GAPPHOST_CULA_SUPPORT

LIBS += -Wl,-E
LIBS += -L${CUDA_LIB_PATH}
LIBS += -L${CULA_LIB_PATH}
LIBS += -L${RESPROG_LIB_PATH}
LIBS += -lboost_iostreams -lcudart -lcula_core -lcula_lapack -lcublas -lz
