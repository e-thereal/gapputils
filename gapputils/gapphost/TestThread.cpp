#include "TestThread.h"

#include <iostream>

void TestThread::run() {
  std::cout << "[" << QThread::currentThreadId() << "] " << "running." << std::endl;
  Q_EMIT progressed();
}

void TestResponder::showProgress() {
  std::cout << "[" << QThread::currentThreadId() << "] " << "showing progress." << std::endl;
}
