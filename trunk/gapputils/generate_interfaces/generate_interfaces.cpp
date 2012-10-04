//============================================================================
// Name        : generate_interfaces.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <capputils/ReflectableClass.h>
#include <capputils/DescriptionAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/FileExistsAttribute.h>
#include <capputils/FlagAttribute.h>
#include <capputils/LibraryLoader.h>
#include <capputils/ReflectableClassFactory.h>

#include <capputils/ArgumentsParser.h>
#include <capputils/Verifier.h>
#include <gapputils/GenerateInterfaceAttribute.h>

#include <boost/units/detail/utility.hpp>

#include <fstream>

class DataModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(DataModel)

  Property(LibraryName, std::string)
  Property(Help, bool)

public:
  DataModel() : _Help(false) { }

};

BeginPropertyDefinitions(DataModel)
  using namespace capputils::attributes;

  DefineProperty(LibraryName, Filename(), FileExists(),
      Description("Name of the library that will be parsed."))
  DefineProperty(Help, Flag(),
      Description("Shows this help."))

EndPropertyDefinitions

void generate_source(const std::string& identifier, const std::string& type, const std::string& header);

int main(int argc, char** argv) {
	using namespace capputils;
	using namespace capputils::reflection;
	using namespace gapputils::attributes;

	DataModel model;
	GenerateInterfaceAttribute* generateInterface;

	ArgumentsParser::Parse(model, argc, argv);
	if (model.getHelp() || !Verifier::Valid(model)) {
	  ArgumentsParser::PrintDefaultUsage(argv[0], model);
	  return 0;
	}

	LibraryLoader& loader = LibraryLoader::getInstance();
	loader.loadLibrary(model.getLibraryName());

	ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
	std::vector<std::string>& names = factory.getClassNames();

	std::cout << "Generating interfaces ..." << std::endl;
	for (unsigned i = 0; i < names.size(); ++i) {
	  boost::shared_ptr<ReflectableClass> object(factory.newInstance(names[i]));
	  std::vector<IClassProperty*>& properties = object->getProperties();
	  for (unsigned iProp = 0; iProp < properties.size(); ++iProp) {
	    if ((generateInterface = properties[iProp]->getAttribute<GenerateInterfaceAttribute>())) {
	      std::cout << "  " << "Generating interface '" << generateInterface->getName() << "' ... " << std::flush;
	      generate_source(generateInterface->getName(),
	          boost::units::detail::demangle(properties[iProp]->getType().name()),
	          generateInterface->getHeader());
	      std::cout << "DONE." << std::endl;
	    }
	  }
	}
	std::cout << "DONE." << std::endl;

	return 0;
}

void generate_source(const std::string& identifier, const std::string& type, const std::string& header) {
  std::ofstream output((identifier + ".cpp").c_str());

  output <<
"#include <gapputils/DefaultWorkflowElement.h>\n"
"#include <capputils/InputAttribute.h>\n"
"#include <capputils/OutputAttribute.h>\n"
"#include <gapputils/InterfaceAttribute.h>\n"
"\n"
"#include <" << header << ">\n"
"\n"
"using namespace capputils::attributes;\n"
"using namespace gapputils::attributes;\n"
"\n"
"namespace interfaces {\n"
"  \n"
"namespace inputs {\n"
"\n"
"class " << identifier << " : public gapputils::workflow::DefaultWorkflowElement<" << identifier << ">\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<" << identifier << ">)\n"
"  WorkflowProperty(Value, Output(\"\"));\n"
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"namespace outputs {\n"
"\n"
"class " << identifier << " : public gapputils::workflow::DefaultWorkflowElement<" << identifier << ">\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<" << identifier << ">)\n"
"  WorkflowProperty(Value, Input(\"\"));\n"
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"}" << std::endl;

  output.close();
}

