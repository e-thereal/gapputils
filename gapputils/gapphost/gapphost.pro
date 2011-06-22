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
		  WorkflowItem.cpp
		  
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
          GenericViewer.h
          
CONFIG += no_keywords debug console
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += /home/tombr/Projects
INCLUDEPATH += /home/tombr/include
INCLUDEPATH += /home/tombr/Programs/cuda/include
INCLUDEPATH += /home/tombr/Programs/cula/include
LIBS += -L/home/tombr/Projects/tinyxml/Debug
LIBS += -L"/home/tombr/Projects/capputils/Debug Shared"
LIBS += -L"/home/tombr/Projects/gapputils/Debug Shared"
LIBS += -L"/home/tombr/Programs/cuda/lib"
LIBS += -L"/home/tombr/Programs/cula/lib"
LIBS += -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem -lcudart -lcula -lcublas
