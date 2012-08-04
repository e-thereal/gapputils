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
          
CONFIG += no_keywords debug console
QMAKE_CXXFLAGS += -std=c++0x -pg
QMAKE_LFLAGS += -pg
INCLUDEPATH += /home/tombr/Projects
INCLUDEPATH += /home/tombr/include
INCLUDEPATH += /res1/software/cuda/include
INCLUDEPATH += /res1/software/cula/include
INCLUDEPATH += /home/tombr/Projects/cmif_v5_3/utilities
DEFINES += GAPPHOST_CULA_SUPPORT
LIBS += -Wl,-E -pg
LIBS += -L/home/tombr/Projects/tinyxml/Debug
LIBS += -L"/home/tombr/Projects/capputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/gapputils/Debug Shared"
LIBS += -L"/res1/software/cuda/lib"
LIBS += -L"/res1/software/cula/lib"
LIBS += -L/home/tombr/lib
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_iostreams -lboost_signals -lboost_filesystem -lcudart -lcula_core -lcula_lapack -lcublas -lz
