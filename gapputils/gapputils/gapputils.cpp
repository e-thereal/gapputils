#include "gapputils.h"

#include "XslTransformation.h"
#include "Compare.h"
#include "CsvReader.h"
#include "CsvWriter.h"
#include "PdfLatex.h"
#include "TextWriter.h"
#include "Concater.h"

namespace gapputils {

void registerClasses() {
  XslTransformation();
  Compare();
  CsvReader();
  CsvWriter();
  PdfLatex();
  TextWriter();
  Concater con;
  con.getSeparator();
}

}
