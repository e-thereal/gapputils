#include "PdfLatex.h"

#include <stdio.h>
#include <sstream>

#include <ObserveAttribute.h>
#include <EventHandler.h>
#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <FileExists.h>
#include <Verifier.h>
#include <FilenameAttribute.h>

using namespace capputils::attributes;
using namespace std;

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(PdfLatex)

  DefineProperty(TexFilename, FileExists(), Input(), Observe(PROPERTY_ID), Filename())
  DefineProperty(CommandName, Observe(PROPERTY_ID))
  DefineProperty(ParameterString, Observe(PROPERTY_ID))
  DefineProperty(OutputName, Output(), Observe(PROPERTY_ID))
  DefineProperty(CommandOutput, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

PdfLatex::PdfLatex(void) : _TexFilename(""), _CommandName("pdflatex"), _ParameterString(" -enable-write18 -interaction=nonstopmode"), _OutputName(""), _CommandOutput("")
{
  this->Changed.connect(capputils::EventHandler<PdfLatex>(this, &PdfLatex::changedHandler));
}

PdfLatex::~PdfLatex(void)
{
}

void PdfLatex::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if ((eventId == 0 || eventId == 1 || eventId == 2) && capputils::Verifier::Valid(*this)) {
    const string& texName = getTexFilename();
    stringstream command;
    stringstream output;
    int ch;

    command << getCommandName().c_str() << " \"" << texName.c_str() << "\" " << getParameterString().c_str();

    output << "Executing: " << command.str() << endl;

    FILE* stream = popen(command.str().c_str(), "r");
    if (stream) {
      while ( (ch=fgetc(stream)) != EOF ) {
        output << (char)ch;
      }
      pclose(stream);
      setOutputName(texName.substr(0, texName.size() - 3) + "pdf");
    } else {
      output << "Error executing command." << endl;
    }

    setCommandOutput(output.str());
  }
}

}
