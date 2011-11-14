#include "PdfLatex.h"

#include <cstdio>
#include <iostream>
#include <sstream>

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <boost/filesystem.hpp>

#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

namespace gapputils {

namespace common {

int PdfLatex::texId;

BeginPropertyDefinitions(PdfLatex)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(TexFilename, ShortName("TeX"), FileExists(), Input(), Observe(texId = PROPERTY_ID), Filename(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(CommandName, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterString, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputName, ShortName("Pdf"), Output(), Observe(PROPERTY_ID), Volatile(), TimeStamp(PROPERTY_ID), Filename())
  DefineProperty(CommandOutput, ShortName("Cout"), Output(), Observe(PROPERTY_ID), Volatile(), Hide(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

PdfLatex::PdfLatex(void) : _TexFilename(""), _CommandName("pdflatex"),
    _ParameterString(" -enable-write18 -interaction=nonstopmode"),
    _OutputName(""), _CommandOutput(""), data(0)
{
  setLabel("PdfLatex");
  Changed.connect(capputils::EventHandler<PdfLatex>(this, &PdfLatex::changedEventHandler));
}

PdfLatex::~PdfLatex(void)
{
  if (data)
    delete data;
}

void PdfLatex::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == texId) {
    const string& texName = getTexFilename();
    setOutputName(texName.substr(0, texName.size() - 3) + "pdf");
  }
}

void PdfLatex::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace boost::filesystem;

  if (!data)
    data = new PdfLatex();

  if (!capputils::Verifier::Valid(*this))
    return;

  stringstream command;
  stringstream output;
  int ch;

  path outputName = getOutputName();
  string outputDir = outputName.branch_path().string();
  path inputName = getTexFilename();
  string texFilename = inputName.leaf().string();
  if (outputDir.size() == 0)
    outputDir = ".";
//  command << getCommandName().c_str() << " -output-directory=\"" << outputDir << "\""
//          << " " << getParameterString().c_str() << " \"" << getTexFilename() << "\"";
  command << "pushd \"" << outputDir << "\" && " << getCommandName().c_str()
          << " " << getParameterString().c_str() << " \"" << texFilename << "\""
          << " && popd";

  cout << "Executing: " << command.str() << endl;
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

void PdfLatex::writeResults() {
  setOutputName(getOutputName());
  setCommandOutput(data->getCommandOutput());
}

}

}
