#include "PdfLatex.h"

#include <stdio.h>
#include <sstream>

#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/Verifier.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/VolatileAttribute.h>
#include "HideAttribute.h"

#include <capputils/ShortNameAttribute.h>

using namespace capputils::attributes;
using namespace std;

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(PdfLatex)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(TexFilename, ShortName("TeX"), FileExists(), Input(), Observe(PROPERTY_ID), Filename(), Volatile())
  DefineProperty(CommandName, Observe(PROPERTY_ID))
  DefineProperty(ParameterString, Observe(PROPERTY_ID))
  DefineProperty(OutputName, ShortName("Pdf"), Output(), Observe(PROPERTY_ID), Volatile())
  DefineProperty(CommandOutput, ShortName("Cout"), Output(), Observe(PROPERTY_ID), Volatile(), Hide())

EndPropertyDefinitions

PdfLatex::PdfLatex(void) : _TexFilename(""), _CommandName("pdflatex"),
    _ParameterString(" -enable-write18 -interaction=nonstopmode"),
    _OutputName(""), _CommandOutput(""), data(0)
{
  setLabel("PdfLatex");
}

PdfLatex::~PdfLatex(void)
{
  if (data)
    delete data;
}

void PdfLatex::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new PdfLatex();

  if (!capputils::Verifier::Valid(*this))
    return;

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
    data->setOutputName(texName.substr(0, texName.size() - 3) + "pdf");
  } else {
    output << "Error executing command." << endl;
  }

  data->setCommandOutput(output.str());
}

void PdfLatex::writeResults() {
  setOutputName(data->getOutputName());
  setCommandOutput(data->getCommandOutput());
}

}
