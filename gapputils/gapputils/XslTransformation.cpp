#include "XslTransformation.h"

#include "InputAttribute.h"
#include "OutputAttribute.h"
#include "LabelAttribute.h"
#include <FileExists.h>
#include <FilenameAttribute.h>
#include <ObserveAttribute.h>
#include <NotEqualAssertion.h>
#include <EventHandler.h>
#include <Verifier.h>
#include "ShortNameAttribute.h"
#include <VolatileAttribute.h>
#include "HideAttribute.h"

#include <cstdio>

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(XslTransformation)

ReflectableBase(workflow::WorkflowElement)
DefineProperty(InputName, ShortName("Xml"), Input(), Filename(), FileExists(), Observe(PROPERTY_ID), Volatile())
DefineProperty(OutputName, ShortName("Out"), Output(), Filename(), Observe(PROPERTY_ID), Volatile())
DefineProperty(XsltName, ShortName("Xslt"), Input(), Filename(), FileExists(), Observe(PROPERTY_ID))
DefineProperty(CommandName, Observe(PROPERTY_ID))
DefineProperty(CommandOutput, ShortName("Cout"), Output(), Observe(PROPERTY_ID), Hide())
DefineProperty(OutputExtension, Observe(PROPERTY_ID))
DefineProperty(InSwitch, Observe(PROPERTY_ID))
DefineProperty(OutSwitch, Observe(PROPERTY_ID))
DefineProperty(XsltSwitch, Observe(PROPERTY_ID))

EndPropertyDefinitions

#ifdef _WIN32
XslTransformation::XslTransformation(void) : _InputName(""), _OutputName(""), _XsltName(""),
  _CommandName("altovaxml"), _CommandOutput(""), _OutputExtension(".xml"),
  _InSwitch("-in "), _OutSwitch("-out "), _XsltSwitch("-xslt2 "), data(0)
{
  setLabel("Xslt");
}
#else
XslTransformation::XslTransformation(void) : _InputName(""), _OutputName(""), _XsltName(""),
  _CommandName("java -jar ~/tools/saxon9he.jar"), _CommandOutput(""), _OutputExtension(".xml"),
  _InSwitch("-s:"), _OutSwitch("-o:"), _XsltSwitch("-xsl:"), data(0)
{
  setLabel("Xslt");
}
#endif

XslTransformation::~XslTransformation(void)
{
  if (data)
    delete data;
}

void XslTransformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new XslTransformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  stringstream command;
  stringstream output;
  int ch;
  string outputName = getInputName() + getOutputExtension();

  command << getCommandName() << " " << getInSwitch() << "\"" << getInputName() << "\" "
          << getOutSwitch() << "\"" << outputName << "\" "
          << getXsltSwitch() << "\"" << getXsltName() << "\"";

  output << "Executing: " << command.str() << endl;

  FILE* stream = popen(command.str().c_str(), "r");
  if (stream) {
    while ( (ch=fgetc(stream)) != EOF ) {
      output << (char)ch;
    }
    pclose(stream);
    data->setOutputName(outputName);
  } else {
    output << "Error executing command." << endl;
  }

  data->setCommandOutput(output.str());
}

void XslTransformation::writeResults() {
  if(!data)
    return;
  setOutputName(data->getOutputName());
  setCommandOutput(data->getCommandOutput());
}

}
