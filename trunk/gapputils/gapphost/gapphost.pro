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
		  ImageViewerItem.cpp \
		  InputsItem.cpp \
		  OutputsItem.cpp \
		  NewObjectDialog.cpp \
		  ShowImageDialog.cpp \
		  DataModel.cpp \
		  Node.cpp \
		  Edge.cpp \
		  Workflow.cpp \
		  TestWorkflow.cpp \
		  Controller.cpp \
		  DefaultInterface.cpp \
		  WorkflowWorker.cpp \
		  GenericViewer.cpp \
		  WorkflowItem.cpp \
		  MakeGlobalDialog.cpp \
		  GlobalProperty.cpp \
		  PopUpList.cpp \
		  GlobalEdge.cpp \
		  EditInterfaceDialog.cpp
		  
HEADERS = MainWindow.h \
          Person.h \
          PropertyReference.h \
          ModelHarmonizer.h \
          PropertyGridDelegate.h \
          Workbench.h \
          GeneratedFiles/ui_NewObjectDialog.h \
          FilenameEdit.h \
          NewObjectDialog.h \
          ShowImageDialog.h \
          Workflow.h \
          WorkflowWorker.h \
          WorkflowItem.h \
          GenericViewer.h \
          MakeGlobalDialog.h \
          PopUpList.h \
          ToolItem.h \
          EditInterfaceDialog.h
          
CONFIG += no_keywords debug console
QMAKE_CXXFLAGS += -std=c++0x -pg
QMAKE_LFLAGS += -pg
INCLUDEPATH += /home/tombr/Projects
INCLUDEPATH += /home/tombr/include
INCLUDEPATH += /res1/software/cuda/include
INCLUDEPATH += /res1/software/cula/include
DEFINES += GAPPHOST_CULA_SUPPORT
LIBS += -Wl,-E -pg
LIBS += -L/home/tombr/Projects/tinyxml/Debug
LIBS += -L"/home/tombr/Projects/capputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/gapputils/Debug Shared"
LIBS += -L"/res1/software/cuda/lib"
LIBS += -L"/res1/software/cula/lib"
LIBS += -L/home/tombr/lib
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem -lcudart -lcula_core -lcula_lapack -lcublas
