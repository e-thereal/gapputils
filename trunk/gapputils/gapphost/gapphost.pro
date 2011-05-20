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
		  GenericViewer.cpp
		  
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
          WorkflowWorker.h
          
CONFIG += no_keywords debug console
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += ../gapputils
INCLUDEPATH += ../capputils
INCLUDEPATH += ../tinyxml
INCLUDEPATH += ../testlib
INCLUDEPATH += /home/tombr/include
LIBS += -L/home/tombr/Projects/tinyxml/Debug -L/home/tombr/Projects/capputils/Debug\ Shared -L"/home/tombr/Projects/gapputils/Debug Shared" -lgapputils -lcapputils -ltinyxml -lboost_signals -lboost_filesystem
