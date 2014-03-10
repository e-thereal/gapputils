//============================================================================
// Name        : generate_interfaces.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <capputils/ArgumentsParser.h>
#include <capputils/GenerateBashCompletion.h>
#include <capputils/LibraryLoader.h>
#include <capputils/Verifier.h>
#include <capputils/Xmlizer.h>

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/FileExistsAttribute.h>
#include <capputils/attributes/FlagAttribute.h>
#include <capputils/attributes/ParameterAttribute.h>
#include <capputils/attributes/OperandAttribute.h>
#include <capputils/attributes/EmptyAttribute.h>
#include <capputils/attributes/OrAttribute.h>

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ReflectableClassFactory.h>

#include <gapputils/attributes/GenerateInterfaceAttribute.h>

#include <boost/units/detail/utility.hpp>

#include <fstream>

#include "InterfaceModel.h"

class DataModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(DataModel)

  Property(LibraryName, std::string)
  Property(OnlyClass, std::string)
  Property(InterfaceModel, std::string)
  Property(GenerateBashCompletion, std::string)
  Property(GenerateTemplate, std::string)
  Property(Verbose, bool)
  Property(Help, bool)

public:
  DataModel() : _Verbose(false), _Help(false) { }

};

BeginPropertyDefinitions(DataModel)
  using namespace capputils::attributes;

  DefineProperty(LibraryName, Filename(), Or(Empty<Type>(), FileExists()),
      Parameter("lib", "l"), Description("Name of the library that will be parsed."))
  DefineProperty(OnlyClass,
      Parameter("class", "c"), Description("Only the class with the given name will be parsed for interfaces. All classes are parsed when empty."))
  DefineProperty(InterfaceModel, Filename(), Or(Empty<Type>(), FileExists()),
      Operand("interfaces"), Description("Name of the XML file containing the description of the interfaces."))
  DefineProperty(GenerateBashCompletion, Filename(),
      Parameter("bash", "b"), Description("Generates a bash_completion configuration file for this program (generate_interfaces)."))
  DefineProperty(GenerateTemplate, Filename(),
      Parameter("template", "t"), Description("Generates an example configuration. Use this as a starting point for your own interface configurations."))
  DefineProperty(Verbose, Flag(),
      Parameter("verbose", "v"), Description("Show more information."))
  DefineProperty(Help, Flag(),
      Parameter("help", "h"), Description("Shows this help."))

EndPropertyDefinitions

void generate_source(gapputils::Interface& interface);
void generate_source(const std::string& identifier, const std::string& type, const std::string& header, bool isParameter, bool isCollection);

int main(int argc, char** argv) {
	using namespace capputils;
	using namespace gapputils;
	using namespace capputils::reflection;
	using namespace gapputils::attributes;

	DataModel model;
	GenerateInterfaceAttribute* generateInterface;

	ArgumentsParser::Parse(model, argc, argv);

  if (model.getHelp() || !Verifier::Valid(model)) {
    ArgumentsParser::PrintDefaultUsage(argv[0], model);
  } else if (model.getGenerateBashCompletion().size()) {
    GenerateBashCompletion::Generate(argv[0], model, model.getGenerateBashCompletion());
  } else if (model.getGenerateTemplate().size()) {
	  InterfaceModel imodel;
	  boost::shared_ptr<Interface> interface(new Interface());
	  interface->setHeader("header.h");
	  interface->setIdentifier("MyType");
	  interface->setType("my::namespace::MyType");
	  interface->setIsParameter(true);
	  interface->setIsCollection(true);
	  imodel.getInterfaces()->push_back(interface);
	  Xmlizer::ToXml(model.getGenerateTemplate(), imodel);
	} else if (model.getInterfaceModel().size()) {
	  InterfaceModel imodel;
	  Xmlizer::FromXml(imodel, model.getInterfaceModel());
	  std::vector<boost::shared_ptr<Interface> >& interfaces = *imodel.getInterfaces();
	  std::cout << "Generating interfaces ..." << std::endl;
	  for (size_t i = 0; i < interfaces.size(); ++i) {
	    std::cout << "  " << "Generating interface '" << interfaces[i]->getIdentifier() << "' ... " << std::flush;
//      generate_source(interfaces[i]->getIdentifier(),
//          interfaces[i]->getType(),
//          interfaces[i]->getHeader(),
//          interfaces[i]->getIsParameter(),
//          interfaces[i]->getIsCollection());
	    generate_source(*interfaces[i]);
      std::cout << "DONE." << std::endl;
	  }
	} else if (model.getLibraryName().size()) {

    LibraryLoader& loader = LibraryLoader::getInstance();
    loader.loadLibrary(model.getLibraryName());

    ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
    std::vector<std::string>& names = factory.getClassNames();

    std::cout << "Generating interfaces ..." << std::endl;
    for (unsigned i = 0; i < names.size(); ++i) {
      if (model.getVerbose())
        std::cout << "  Found class: '" << names[i] << "'" << std::flush;
      if (model.getOnlyClass().size() && model.getOnlyClass() != names[i]) {
        if (model.getVerbose())
          std::cout << " SKIPPED!" << std::endl;
        continue;
      } else if (model.getVerbose()) {
        std::cout << std::endl;
      }
      boost::shared_ptr<ReflectableClass> object(factory.newInstance(names[i]));
      std::vector<IClassProperty*>& properties = object->getProperties();
      for (unsigned iProp = 0; iProp < properties.size(); ++iProp) {
        if ((generateInterface = properties[iProp]->getAttribute<GenerateInterfaceAttribute>())) {
          std::cout << "  " << "Generating interface '" << generateInterface->getName() << "' ... " << std::flush;
          generate_source(generateInterface->getName(),
              boost::units::detail::demangle(properties[iProp]->getType().name()),
              generateInterface->getHeader(),
              generateInterface->getIsParameter(), false);
          std::cout << "DONE." << std::endl;
        }
      }
    }
    std::cout << "DONE." << std::endl;
	} else {
	  ArgumentsParser::PrintDefaultUsage(argv[0], model);
	}

	return 0;
}

void generate_source(gapputils::Interface& interface) {
  std::string identifier = interface.getIdentifier();
  std::string header = interface.getHeader();
  std::string type = interface.getType();
  bool isCollection = interface.getIsCollection();
  bool isParameter = interface.getIsParameter();
  std::vector<std::string>& attributes = *interface.getPropertyAttributes();

  std::ofstream output((identifier + (isParameter ? "_parameter.cpp" : "_interface.cpp")).c_str());

  std::string baseType;
  if (isCollection)
    baseType = "gapputils::workflow::CollectionElement";
  else
    baseType = "gapputils::workflow::DefaultWorkflowElement<" + identifier + ">";

  output <<
"#include <gapputils/DefaultWorkflowElement.h>\n"
"#include <gapputils/CollectionElement.h>\n"
"\n"
"#include <capputils/attributes/InputAttribute.h>\n"
"#include <capputils/attributes/OutputAttribute.h>\n"
"#include <gapputils/attributes/InterfaceAttribute.h>\n"
"\n";

  if (header.size()) {
    output <<
"#include \"" << header << "\"\n"
"\n";
  }
  output <<
"using namespace capputils::attributes;\n"
"using namespace gapputils::attributes;\n"
"\n";
  if (isParameter) {
    output <<
"namespace interfaces {\n"
"  \n"
"namespace parameters {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, std::vector<property_t>)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Enumerable<Type, false>())\n"
"  WorkflowProperty(Value, FromEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value";
      for (size_t i = 0; i < attributes.size(); ++i)
        output << ", " << attributes[i];
      output << ");\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"}" << std::endl;
  } else {
    output <<
"namespace interfaces {\n"
"  \n"
"namespace inputs {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, boost::shared_ptr<std::vector<property_t> >)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Output(\"Values\"), Enumerable<Type, false>(), NotNull<Type>())\n"
"  WorkflowProperty(Value, Output(\"Value\"), FromEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value, Output(\"\"));\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"namespace outputs {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, boost::shared_ptr<std::vector<property_t> >)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Input(\"Values\"), Enumerable<Type, false>())\n"
"  WorkflowProperty(Value, Input(\"Value\"), ToEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value, Input(\"\"));\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"}" << std::endl;
  }
  output.close();
}

void generate_source(const std::string& identifier, const std::string& type, const std::string& header, bool isParameter, bool isCollection) {
  std::ofstream output((identifier + (isParameter ? "_parameter.cpp" : ".cpp")).c_str());

  std::string baseType;
  if (isCollection)
    baseType = "gapputils::workflow::CollectionElement";
  else
    baseType = "gapputils::workflow::DefaultWorkflowElement<" + identifier + ">";

  output <<
"#include <gapputils/DefaultWorkflowElement.h>\n"
"#include <gapputils/CollectionElement.h>\n"
"\n"
"#include <capputils/attributes/InputAttribute.h>\n"
"#include <capputils/attributes/OutputAttribute.h>\n"
"#include <gapputils/attributes/InterfaceAttribute.h>\n"
"\n";

  if (header.size()) {
    output <<
"#include <" << header << ">\n"
"\n";
  }
  output <<
"using namespace capputils::attributes;\n"
"using namespace gapputils::attributes;\n"
"\n";
  if (isParameter) {
    output <<
"namespace interfaces {\n"
"  \n"
"namespace parameters {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, std::vector<property_t>)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "() { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Enumerable<Type, false>())\n"
"  WorkflowProperty(Value, FromEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value);\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"}" << std::endl;
  } else {
    output <<
"namespace interfaces {\n"
"  \n"
"namespace inputs {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, boost::shared_ptr<std::vector<property_t> >)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "()" << (isCollection ? " : _Values(new std::vector<property_t>())" : "") << " { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Output(\"Values\"), Enumerable<Type, false>(), NotNull<Type>())\n"
"  WorkflowProperty(Value, Output(\"Value\"), FromEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value, Output(\"\"));\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"namespace outputs {\n"
"\n"
"class " << identifier << " : public " << baseType << "\n"
"{\n"
"  InitReflectableClass(" << identifier << ")\n"
"  \n"
"  typedef " << type << " property_t;\n"
"  \n" <<
(isCollection ? "  Property(Values, boost::shared_ptr<std::vector<property_t> >)\n" : "") <<
"  Property(Description, std::string)\n"
"  Property(Value, property_t)\n"
"  \n"
"public:\n"
"  " << identifier << "()" << (isCollection ? " : _Values(new std::vector<property_t>())" : "") << " { setLabel(\"" << identifier << "\"); }\n"
"};\n"
"\n"
"BeginPropertyDefinitions(" << identifier << ", Interface())\n"
"  ReflectableBase(" << baseType << ")\n"
"  WorkflowProperty(Description)\n";
    if (isCollection) {
      output <<
"  WorkflowProperty(Values, Input(\"Values\"), Enumerable<Type, false>())\n"
"  WorkflowProperty(Value, Input(\"Value\"), ToEnumerable(Id - 1));\n";
    } else {
      output <<
"  WorkflowProperty(Value, Input(\"\"));\n";
    }
    output <<
"EndPropertyDefinitions\n"
"\n"
"}\n"
"\n"
"}" << std::endl;
  }
  output.close();
}

