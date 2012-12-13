/*
 * AverageMifs.h
 *
 *  Created on: Dec 11, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_AVERAGEMIFS_H_
#define GAPPUTILS_CV_AVERAGEMIFS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gapputils {
namespace cv {

class AverageMifs : public DefaultWorkflowElement<AverageMifs> {

  InitReflectableClass(AverageMifs)

  Property(MifNames, std::vector<std::string>)
  Property(OutputName, std::string)
  Property(Output, std::string)

public:
  AverageMifs();
  virtual ~AverageMifs();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cv */
} /* namespace gapputils */
#endif /* GAPPUTILS_CV_AVERAGEMIFS_H_ */
