/*
 * Encoder.h
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#ifndef GML_ENCODER_H_
#define GML_ENCODER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"
#include "CodingDirection.h"

namespace gml {

namespace rbm {

struct EncoderChecker { EncoderChecker(); };

/**
 * \brief Encodes a set of feature vectors using an RBM
 *
 * VisibleSet is an array of size n times RbmModel.getVisibleCount
 * A new set HiddenSet is returned of size n times RbmModel.getHiddenCount
 * where n is the number of feature vectors
 *
 * Encoding just a single feature vector is natural special case with a set
 * of vectors of size 1 (the visible set contains just one single vector)
 *
 * The mean and the standard deviation of each visible feature is used from
 * the RbmModel to perform feature scaling
 *
 * Currently, the hidden unit is replaced by the expectation rather than
 * sampling from its distribution.
 */
class Encoder : public DefaultWorkflowElement<Encoder> {

  typedef model_t::value_t value_t;
  typedef model_t::host_matrix_t host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef matrix_t::dim_t dim_t;

  typedef boost::shared_ptr<std::vector<double> > data_t;

  friend class EncoderChecker;

  InitReflectableClass(Encoder)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<std::vector<data_t> >)
  Property(Direction, CodingDirection)
  Property(DoubleWeights, bool)
  Property(OnlyFilters, bool)
  Property(NormalizeOnly, bool)
  Property(Outputs, boost::shared_ptr<std::vector<data_t> >)

public:
  Encoder();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}


#endif /* GML_ENCODER_H_ */
