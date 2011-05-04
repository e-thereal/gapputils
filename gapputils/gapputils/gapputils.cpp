#include "gapputils.h"

#include "AltovaXml.h"
#include "Compare.h"
#include "CsvReader.h"
#include "CsvWriter.h"
#include "PdfLatex.h"
#include "TextWriter.h"

namespace gapputils {

void registerClasses() {
  AltovaXml();
  Compare();
  CsvReader();
  CsvWriter();
  PdfLatex();
  TextWriter();
}

}
