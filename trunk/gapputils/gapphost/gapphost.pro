SOURCES = $$files(interfaces/*.cpp) \ 
		  main.cpp \
		  MainWindow.cpp \
		  PropertyReference.cpp \
		  ModelHarmonizer.cpp \
		  PropertyGridDelegate.cpp \
		  Workbench.cpp \
		  ToolItem.cpp \
		  CableItem.cpp \
		  FilenameEdit.cpp \
		  ShowImageDialog.cpp \
		  DataModel.cpp \
		  Node.cpp \
		  Edge.cpp \
		  Workflow.cpp \
		  Controller.cpp \
		  WorkflowItem.cpp \
		  LineEditDialog.cpp \
		  GlobalProperty.cpp \
		  PopUpList.cpp \
		  GlobalEdge.cpp \
		  Expression.cpp \
		  HostInterface.cpp \
		  linreg.cpp \
		  ChecksumUpdater.cpp \
		  WorkflowController.cpp \
		  WorkflowUpdater.cpp \
		  NodeCache.cpp \
		  WorkflowToolBox.cpp \
		  PropertyGrid.cpp \
		  LogbookModel.cpp \
		  LogbookWidget.cpp \
		  GlobalPropertiesView.cpp \
		  WorkbenchWindow.cpp \
		  TestModule.cpp \
		  HeadlessApp.cpp \
		  WorkflowSnippets.cpp \
		  MergeTest.cpp \
		  HorizontalAnnotation.cpp \
		  VerticalAnnotation.cpp \
		  MemoryTest.cpp \
		  ModuleHelpWidget.cpp \
		  GlobalPropertiesViewDelegate.cpp \
		  MessageBox.cpp
		  
HEADERS = MainWindow.h \
          PropertyReference.h \
          ModelHarmonizer.h \
          PropertyGridDelegate.h \
          Workbench.h \
          FilenameEdit.h \
          ShowImageDialog.h \
          Workflow.h \
          WorkflowItem.h \
          GenericViewer.h \
          LineEditDialog.h \
          PopUpList.h \
          ToolItem.h \
          WorkflowUpdater.h \
          WorkflowToolBox.h \
          PropertyGrid.h \
          LogbookModel.h \
          LogbookWidget.h \
          GlobalPropertiesView.h \
          WorkbenchWindow.h \
          HeadlessApp.h \
          WorkflowSnippets.h \
          ModuleHelpWidget.h \
          GlobalPropertiesViewDelegate.h
          
RESOURCES = res.qrc
          
CONFIG += no_keywords console

CONFIG(debug, debug|release) {
  DEFINES += _DEBUG
  TARGET = grapevined

  QMAKE_CXXFLAGS += -pg
  QMAKE_LFLAGS += -pg
  LIBS += -pg
  LIBS += -L"../../tinyxml/Debug"
  LIBS += -L"../../capputils/Debug"
  LIBS += -L"../../gapputils/Debug"
  LIBS += -lgapputilsd -lcapputilsd -ltinyxmld
  message("Debug build")
}

CONFIG(release, debug|release) {
  DEFINES += _RELEASE
  TARGET = grapevine

  LIBS += -L"../../tinyxml/Release"
  LIBS += -L"../../capputils/Release"
  LIBS += -L"../../gapputils/Release"
  LIBS += -lgapputils -lcapputils -ltinyxml
  message("Release build")
}

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += ".."
INCLUDEPATH += ${RESPROG_INC_PATH}

LIBS += -Wl,-E
LIBS += -L${RESPROG_LIB_PATH}
LIBS += -lboost_iostreams -lboost_filesystem -lboost_signals -lboost_system -lboost_regex -lz