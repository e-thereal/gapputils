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
		  NewObjectDialog.cpp \
		  ShowImageDialog.cpp \
		  DataModel.cpp \
		  Node.cpp \
		  Edge.cpp \
		  WorkflowController.cpp \
		  Graph.cpp
		  
		  
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
          WorkflowController.h
          
CONFIG += no_keywords debug
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += ../gapputils
INCLUDEPATH += ../capputils
INCLUDEPATH += ../tinyxml
INCLUDEPATH += /home/tombr/include
LIBS += -L/home/tombr/Projects/tinyxml/Debug -L/home/tombr/Projects/capputils/Debug -L/home/tombr/Projects/gapputils/Debug -lgapputils -lcapputils -ltinyxml -lboost_signals
