/*
 * HeadlessApp.cpp
 *
 *  Created on: Dec 17, 2012
 *      Author: tombr
 */

#include "HeadlessApp.h"

#include <qcoreapplication.h>

#include <capputils/Logbook.h>

#include "DataModel.h"
#include "Workflow.h"
#include "WorkflowUpdater.h"

using namespace capputils;
using namespace gapputils::workflow;

namespace gapputils {

namespace host {

HeadlessApp::HeadlessApp(QWidget *parent) : QObject(parent), workflowUpdater(new WorkflowUpdater()) {
  connect(workflowUpdater.get(), SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  connect(workflowUpdater.get(), SIGNAL(progressed(boost::shared_ptr<workflow::Node>, double)), this, SLOT(showProgress(boost::shared_ptr<workflow::Node>, double)));
}

HeadlessApp::~HeadlessApp() {
}

void HeadlessApp::resume() {
  DataModel& model = DataModel::getInstance();

  boost::shared_ptr<Workflow> workflow = model.getMainWorkflow();

//  grandpa = boost::make_shared<workflow::Workflow>();
//  grandpa->getNodes()->push_back(workflow);
//  grandpa->resume();
  workflow->resume();
}

bool HeadlessApp::updateMainWorkflow() {
  DataModel& model = DataModel::getInstance();
  workflowUpdater->update(model.getMainWorkflow());

  return true;
}

bool HeadlessApp::updateMainWorkflowNode(const std::string& label) {
  DataModel& model = DataModel::getInstance();
  boost::shared_ptr<Workflow> workflow = model.getMainWorkflow();
  Logbook& dlog = *workflow->getLogbook();

  boost::shared_ptr<Node> node = workflow->getNodeByLabel(label);
  if(!node) {
    dlog(Severity::Error) << "Could not find node with the label '" << label << "'. Won't update workflow.";
    return false;
  }

  workflowUpdater->update(node);
  return true;
}

bool HeadlessApp::updateMainWorkflowNodes(const std::vector<std::string>& labels) {
  DataModel& model = DataModel::getInstance();
  boost::shared_ptr<Workflow> workflow = model.getMainWorkflow();
  Logbook& dlog = *workflow->getLogbook();

  boost::shared_ptr<std::vector<boost::shared_ptr<Node> > > nodes(new std::vector<boost::shared_ptr<Node> >());

  for (size_t i = 0; i < labels.size(); ++i) {
    boost::shared_ptr<Node> node = workflow->getNodeByLabel(labels[i]);
    if(!node) {
      dlog(Severity::Error) << "Could not find node with the label '" << labels[i] << "'. Won't update workflow.";
      return false;
    }
    nodes->push_back(node);
  }

  workflowUpdater->update(nodes);
  return true;
}

void HeadlessApp::updateFinished() {
  QCoreApplication::exit(0);
}

void HeadlessApp::showProgress(boost::shared_ptr<workflow::Node>, double) {

}

} /* namespace host */
} /* namespace gapputils */
