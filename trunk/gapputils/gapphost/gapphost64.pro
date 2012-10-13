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

DEFINES += GAPPHOST_CULA_SUPPORT
          
CONFIG += no_keywords release console
QMAKE_CXXFLAGS += -std=c++0x -fPIC
INCLUDEPATH += /res1/software/usr/include
INCLUDEPATH += /res1/software/x64/cuda/include
INCLUDEPATH += /res1/software/x64/cula/include
LIBS += -Wl,-E
LIBS += -L"/res1/software/usr/lib64"
LIBS += -L"/res1/software/x64/cuda/lib64"
LIBS += -L"/res1/software/x64/cula/lib64"
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_iostreams -lboost_signals -lboost_filesystem -lboost_system -lcudart -lcula_core -lcula_lapack -lcublas -lz
