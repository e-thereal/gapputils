#include "TextWriter.h"

#include <fstream>

#include <capputils/InputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/EventHandler.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(TextWriter)

ReflectableBase(workflow::WorkflowElement)
DefineProperty(Text, Input(), Observe(PROPERTY_ID), Volatile(), TimeStamp(PROPERTY_ID))
DefineProperty(Filename, Filename(), NotEqual<string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

TextWriter::TextWriter() : _Text(""), _Filename("") {
  setLabel("Writer");
}

TextWriter::~TextWriter() { }

void TextWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (capputils::Verifier::Valid(*this)) {
    ofstream outfile(getFilename().c_str());
    outfile << getText();
    outfile.close();
  }
}

void TextWriter::writeResults() {
}

}

}
