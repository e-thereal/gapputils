#include "TextWriter.h"

#include <fstream>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/EventHandler.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/HideAttribute.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

int TextWriter::inputId;

BeginPropertyDefinitions(TextWriter)

ReflectableBase(workflow::WorkflowElement)
DefineProperty(Text, Input(), Observe(inputId = PROPERTY_ID), Volatile(), Hide(), TimeStamp(PROPERTY_ID))
DefineProperty(Filename, Output(""), Filename(), NotEqual<string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
DefineProperty(Auto, Observe(PROPERTY_ID))

EndPropertyDefinitions

TextWriter::TextWriter() : _Text(""), _Filename(""), _Auto(false) {
  setLabel("Writer");
  Changed.connect(capputils::EventHandler<TextWriter>(this, &TextWriter::changedHandler));
}

TextWriter::~TextWriter() { }

void TextWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
  }

  /*std::cout << getProperties()[eventId]->getName() << " changed to "
            << getProperties()[eventId]->getStringValue(*this) << std::endl;*/
}

void TextWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (capputils::Verifier::Valid(*this)) {
    std::cout << "[Info] Writing text file: " << getFilename() << std::endl;
    ofstream outfile(getFilename().c_str());
    outfile << getText();
    outfile.close();
  }
}

void TextWriter::writeResults() {
}

}

}
