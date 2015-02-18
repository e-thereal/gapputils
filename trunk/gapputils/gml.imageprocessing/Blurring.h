#pragma once
#ifndef GML_BLURRING_H_
#define GML_BLURRING_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

struct BlurringChecker { BlurringChecker(); };

class Blurring : public DefaultWorkflowElement<Blurring> {

  friend struct BlurringChecker;

  InitReflectableClass(Blurring)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(SigmaX, double)
  Property(SigmaY, double)
  Property(SigmaZ, double)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  Blurring(void);
  
protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_BLURRING_H_ */
