#define BOOST_FILESYSTEM_VERSION 2

#include "MainWindow.h"
#include <QtGui/QApplication>
#include <qdir.h>

// TODO: do the cuda, cublas and cula initialization stuff only if requested
#include <cublas.h>
#ifdef GAPPHOST_CULA_SUPPORT
#include <cula.h>
#endif

#include <capputils/Xmlizer.h>
#include <capputils/ArgumentsParser.h>
#include <capputils/Verifier.h>
#include <iostream>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/FactoryException.h>
#include <sstream>

#include "DataModel.h"
#include "Workflow.h"
#include "DefaultInterface.h"
#include "LogbookModel.h"
#include <capputils/Logbook.h>

#include <memory>

//#include <CProcessInfo.hpp>

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;

#include <culib/lintrans.h>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <exception>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "gapphost.h"







#include "client.h"
#include "messagesessionhandler.h"
#include "messageeventhandler.h"
#include "messageeventfilter.h"
#include "chatstatehandler.h"
#include "chatstatefilter.h"
#include "connectionlistener.h"
#include "disco.h"
#include "message.h"
#include "gloox.h"
#include "lastactivity.h"
#include "loghandler.h"
#include "logsink.h"
#include "connectiontcpclient.h"
#include "connectionsocks5proxy.h"
#include "connectionhttpproxy.h"
#include "messagehandler.h"
using namespace gloox;

#include <stdio.h>
#include <string>

#include <cstdio> // [s]print[f]

#if defined( WIN32 ) || defined( _WIN32 )
# include <windows.h>
#endif

// create your own wrapper around this here. Will use qthread and send signals. Maybe not.
class MessageTest : public MessageSessionHandler, ConnectionListener, MessageHandler
{
  public:
    MessageTest() : m_session( 0 ), connected(false) {}

    virtual ~MessageTest() {}

    void start() {
      JID jid( "gapphost@gmail.com/gloox" );
      j = new Client( jid, "powergrapevinepinkpower" );
      j->registerConnectionListener(this);
      j->registerMessageSessionHandler(this, 0);
      if( j->connect( false ) )
      {
        ConnectionError ce = ConnNoError;
        while( ce == ConnNoError )
        {
          ce = j->recv();
        }
        printf( "ce: %d\n", ce );
      }

      delete( j );
    }

    virtual void onConnect()
    {
      printf( "connected!!!\n" );
      connected = true;
    }

    virtual void onDisconnect( ConnectionError e )
    {
      connected = false;
      printf( "message_test: disconnected: %d\n", e );
      if( e == ConnAuthenticationFailed )
        printf( "auth failed. reason: %d\n", j->authError() );
    }

    virtual bool onTLSConnect( const CertInfo& info )
    {
      time_t from( info.date_from );
      time_t to( info.date_to );

      printf( "status: %d\nissuer: %s\npeer: %s\nprotocol: %s\nmac: %s\ncipher: %s\ncompression: %s\n"
              "from: %s\nto: %s\n",
              info.status, info.issuer.c_str(), info.server.c_str(),
              info.protocol.c_str(), info.mac.c_str(), info.cipher.c_str(),
              info.compression.c_str(), ctime( &from ), ctime( &to ) );
      return true;
    }

    virtual void handleMessage( const Message& msg, MessageSession * /*session*/ )
    {
      printf( "type: %d, subject: %s, message: %s, thread id: %s\n", msg.subtype(),
              msg.subject().c_str(), msg.body().c_str(), msg.thread().c_str() );

      std::string re = "You said:\n> " + msg.body() + "\nI like that statement.";
      Sleep(1000);
      m_session->send(re);

      if( msg.body() == "quit" )
        j->disconnect();
    }

    virtual void handleMessageSession(MessageSession *session) {
      printf( "got new session\n");
      // this example can handle only one session. so we get rid of the old session
      j->disposeMessageSession(m_session);
      m_session = session;
      m_session->registerMessageHandler(this);
    }

  private:
    Client *j;
    MessageSession *m_session;
    bool connected;
};


int main(int argc, char *argv[])
{
  MessageTest *r = new MessageTest();
  r->start();
  delete(r);
  return 0;

  qRegisterMetaType<std::string>("std::string");

  QCoreApplication::setOrganizationName("gapputils");
  QCoreApplication::setOrganizationDomain("gapputils.blogspot.com");
  QCoreApplication::setApplicationName("grapevine");

  cublasInit();
  Logbook dlog(&LogbookModel::GetInstance());
  dlog.setModule("host");

  //MSMRI::CProcessInfo::getInstance().getCommandLine(argc, argv);

  boost::filesystem::create_directories(".gapphost");
  boost::filesystem::create_directories(DataModel::getConfigurationDirectory());

#ifdef GAPPHOST_CULA_SUPPORT
  culaStatus status;

  if ((status = culaInitialize()) != culaNoError) {
    std::cout << "Could not initialize CULA: " << culaGetStatusString(status) << std::endl;
    return 1;
  }
#endif

  int ret = 0;
  QApplication a(argc, argv);
  DataModel& model = DataModel::getInstance();
  ArgumentsParser::Parse(model, argc, argv);      // need to be here to read the configuration filename
  try {
    Xmlizer::FromXml(model, DataModel::getConfigurationDirectory() + "/config.xml");
    Xmlizer::FromXml(model, "gapphost.conf.xml"); // compatibility to old versions
    Xmlizer::FromXml(model, model.getConfiguration());
  } catch (capputils::exceptions::FactoryException ex) {
    cout << ex.what() << endl;
    return 1;
  }

  // Initialize if necessary
  if (!model.getMainWorkflow()) {
    boost::shared_ptr<Workflow> workflow(new Workflow());
    model.setMainWorkflow(workflow);
    //model.setCurrentWorkflow(workflow->getUuid());
  }
  if (!model.getMainWorkflow()->getModule())
    model.getMainWorkflow()->setModule(boost::shared_ptr<DefaultInterface>(new DefaultInterface()));

  reflection::ReflectableClass& wfModule = *model.getMainWorkflow()->getModule();

  ArgumentsParser::Parse(model, argc, argv);    // Needs to be here again to override configuration file parameters
  ArgumentsParser::Parse(wfModule, argc, argv);
  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage("gapphost", model);
    ArgumentsParser::PrintUsage("Workflow switches:", wfModule);

    cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
    culaShutdown();
#endif
    return 0;
  }

  try {
    MainWindow w;
    w.show();
    dlog() << "Start resuming ...";
    w.resume();
    //dlog() << "[Info] Resuming done.";
    if (model.getRun()) {
      w.setAutoQuit(true);
      //std::cout << "[Info] Update main workflow." << std::endl;
      w.updateMainWorkflow();
    }
    //std::cout << "[Info] Entering event loop." << std::endl;
    ret = a.exec();
    //std::cout << "[Info] Quitting." << std::endl;
  } catch (char const* error) {
    cout << error << endl;
    return 1;
  }

  model.save();
  model.setMainWorkflow(boost::shared_ptr<Workflow>());
//  delete model.getMainWorkflow();

  cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
  culaShutdown();
#endif
  return ret;
}
