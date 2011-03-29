SOURCES = main.cpp \
		  MainWindow.cpp \
		  Person.cpp \
		  PropertyReference.cpp \
		  ModelHarmonizer.cpp \
		  PropertyGridDelegate.cpp \
		  Workbench.cpp \
		  ToolItem.cpp \
		  LabelAttribute.cpp \
		  CableItem.cpp \
		  FilenameEdit.cpp \
		  ImageLoader.cpp \
		  ImageViewer.cpp \
		  ImageViewerItem.cpp \
		  InputAttribute.cpp \
		  NewObjectDialog.cpp \
		  OutputAttribute.cpp \
		  ShowImageDialog.cpp
		  
HEADERS = MainWindow.h \
          Person.h \
          PropertyReference.h \
          ModelHarmonizer.h \
          PropertyGridDelegate.h \
          Workbench.h \
          LabelAttribute.h \
          GeneratedFiles/ui_NewObjectDialog.h \
          FilenameEdit.h \
          NewObjectDialog.h \
          ShowImageDialog.h
          
          
CONFIG += no_keywords debug
QMAKE_CXXFLAGS += -std=c++0x
INCLUDEPATH += ../capputils
INCLUDEPATH += ../tinyxml
LIBS += -L/home/tombr/Projects/tinyxml/Debug -L/home/tombr/Projects/capputils/Debug -lcapputils -ltinyxml -lboost_signals