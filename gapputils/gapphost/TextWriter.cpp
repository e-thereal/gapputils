#include "TextWriter.h"

#include <fstream>

#include <InputAttribute.h>
#include <FilenameAttribute.h>
#include <ObserveAttribute.h>
#include <NotEqualAssertion.h>
#include <EventHandler.h>
#include <Verifier.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(TextWriter)

DefineProperty(Text, Input(), Observe(PROPERTY_ID))
DefineProperty(Filename, Filename(), NotEqual<string>(""), Observe(PROPERTY_ID))

EndPropertyDefinitions

TextWriter::TextWriter() : _Text(""), _Filename("") {
  Changed.connect(capputils::EventHandler<TextWriter>(this, &TextWriter::changeHandler));
}

TextWriter::~TextWriter() { }

void TextWriter::changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (capputils::Verifier::Valid(*this)) {
    ofstream outfile(getFilename().c_str());
    outfile << getText();
    outfile.close();
  }
}

}