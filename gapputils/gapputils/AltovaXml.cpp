#include "AltovaXml.h"

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

#include <cstdio>

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

enum PropertyIds {
  InputId, OutputId, XsltId, CommandId, CommandOutputId
};

BeginPropertyDefinitions(AltovaXml)

DefineProperty(InputName, ShortName("Xml"), Input(), Filename(), FileExists(), Observe(InputId))
DefineProperty(OutputName, ShortName("Out"), Output(), Filename(), NotEqual<string>(""), Observe(OutputId))
DefineProperty(XsltName, ShortName("Xslt"), Input(), Filename(), FileExists(), Observe(XsltId))
DefineProperty(CommandName, Observe(CommandId))
DefineProperty(CommandOutput, ShortName("Cout"), Output(), Observe(CommandOutputId))

EndPropertyDefinitions

AltovaXml::AltovaXml(void) : _InputName(""), _OutputName("gp.tex"), _XsltName("gp.xslt"), _CommandName("altovaxml"), _CommandOutput("")
{
  Changed.connect(capputils::EventHandler<AltovaXml>(this, &AltovaXml::changeHandler));
}


AltovaXml::~AltovaXml(void)
{
}

void AltovaXml::changeHandler(capputils::ObservableClass* sender, int eventId) {

  // TODO: rethink the trigger. It could be wanted to redo it, when the outputname has changed
  if (eventId != CommandOutputId && eventId != OutputId && capputils::Verifier::Valid(*this)) {
    stringstream command;
    stringstream output;
    int ch;

    command << getCommandName() << " -in \"" << getInputName() << "\" -out \"" << getOutputName() << "\" -xslt2 \"" << getXsltName() << "\"";

    output << "Executing: " << command.str() << endl;

    FILE* stream = popen(command.str().c_str(), "r");
    if (stream) {
      while ( (ch=fgetc(stream)) != EOF ) {
        output << (char)ch;
      }
      pclose(stream);
      setOutputName(getOutputName());
    } else {
      output << "Error executing command." << endl;
    }

    setCommandOutput(output.str());
  }
}

}
