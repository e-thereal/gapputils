#include "XslTransformation.h"

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/EventHandler.h>
#include <capputils/Verifier.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <capputils/HideAttribute.h>
#include <gapputils/LabelAttribute.h>

#include <cstdio>

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

int XslTransformation::inputId;
int XslTransformation::extId;

BeginPropertyDefinitions(XslTransformation)

ReflectableBase(workflow::WorkflowElement)
DefineProperty(InputName, ShortName("Xml"), Input(), Filename(), FileExists(), Observe(inputId = Id), TimeStamp(Id))
DefineProperty(OutputName, ShortName("Out"), Output(), Filename(), Observe(Id), TimeStamp(Id))
DefineProperty(XsltName, ShortName("Xslt"), Input(), Filename(), FileExists(), Observe(Id), TimeStamp(Id))
DefineProperty(CommandName, Observe(Id), TimeStamp(Id))
DefineProperty(CommandOutput, ShortName("Cout"), Output(), Observe(Id), Hide(), TimeStamp(Id))
DefineProperty(OutputExtension, Observe(extId = Id), TimeStamp(Id))
DefineProperty(InSwitch, Observe(Id), TimeStamp(Id))
DefineProperty(OutSwitch, Observe(Id), TimeStamp(Id))
DefineProperty(XsltSwitch, Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

#ifdef _WIN32
XslTransformation::XslTransformation(void) : _InputName(""), _OutputName(""), _XsltName(""),
  _CommandName("altovaxml"), _CommandOutput(""), _OutputExtension(".xml"),
  _InSwitch("-in "), _OutSwitch("-out "), _XsltSwitch("-xslt2 "), data(0)
{
  setLabel("Xslt");
  Changed.connect(capputils::EventHandler<XslTransformation>(this, &XslTransformation::changedEventHandler));
}
#else
XslTransformation::XslTransformation(void) : _InputName(""), _OutputName(""), _XsltName(""),
  _CommandName("java -jar ~/tools/saxon9he.jar"), _CommandOutput(""), _OutputExtension(".xml"),
  _InSwitch("-s:"), _OutSwitch("-o:"), _XsltSwitch("-xsl:"), data(0)
{
  setLabel("Xslt");
  Changed.connect(capputils::EventHandler<XslTransformation>(this, &XslTransformation::changedEventHandler));
}
#endif

XslTransformation::~XslTransformation(void)
{
  if (data)
    delete data;
}

void XslTransformation::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId || eventId == extId) {
    setOutputName(getInputName() + getOutputExtension());
  }
}

void XslTransformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new XslTransformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  stringstream command;
  stringstream output;
  int ch;

  command << getCommandName() << " " << getInSwitch() << "\"" << getInputName() << "\" "
          << getOutSwitch() << "\"" << getOutputName() << "\" "
          << getXsltSwitch() << "\"" << getXsltName() << "\"";

  output << "Executing: " << command.str() << endl;

  FILE* stream = popen(command.str().c_str(), "r");
  if (stream) {
    while ( (ch=fgetc(stream)) != EOF ) {
      output << (char)ch;
    }
    pclose(stream);
  } else {
    output << "Error executing command." << endl;
  }

  data->setCommandOutput(output.str());
}

void XslTransformation::writeResults() {
  if(!data)
    return;
  setOutputName(getOutputName());
  setCommandOutput(data->getCommandOutput());
}

}

}
