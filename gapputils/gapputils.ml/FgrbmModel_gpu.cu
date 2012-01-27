/*
 * FgrbmModel_gpu.cu
 *
 *  Created on: Jan 26, 2012
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "FgrbmModel.h"

#include <thrust/copy.h>

namespace gapputils {

namespace ml {

/*
  Property(VisibleBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(HiddenBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(VisibleWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(HiddenWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(ConditionalWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(VisibleMean, double)
  Property(VisibleStd, double)
  Property(IsGaussian, bool)
 */

boost::shared_ptr<FgrbmModel> FgrbmModel::clone() {
  boost::shared_ptr<FgrbmModel> model(new FgrbmModel());

  boost::shared_ptr<tbblas::device_vector<double> > vb(new tbblas::device_vector<double>(getVisibleBiases()->size()));
  boost::shared_ptr<tbblas::device_vector<double> > hb(new tbblas::device_vector<double>(getHiddenBiases()->size()));
  boost::shared_ptr<tbblas::device_matrix<double> > vw(new tbblas::device_matrix<double>(getVisibleWeights()->size1(), getVisibleWeights()->size2()));
  boost::shared_ptr<tbblas::device_matrix<double> > hw(new tbblas::device_matrix<double>(getHiddenWeights()->size1(), getHiddenWeights()->size2()));
  boost::shared_ptr<tbblas::device_matrix<double> > cw(new tbblas::device_matrix<double>(getConditionalWeights()->size1(), getConditionalWeights()->size2()));

  thrust::copy(getVisibleBiases()->data().begin(), getVisibleBiases()->data().end(), vb->data().begin());
  thrust::copy(getHiddenBiases()->data().begin(), getHiddenBiases()->data().end(), hb->data().begin());
  thrust::copy(getVisibleWeights()->data().begin(), getVisibleWeights()->data().end(), vw->data().begin());
  thrust::copy(getHiddenWeights()->data().begin(), getHiddenWeights()->data().end(), hw->data().begin());
  thrust::copy(getConditionalWeights()->data().begin(), getConditionalWeights()->data().end(), cw->data().begin());

  model->setVisibleBiases(vb);
  model->setHiddenBiases(hb);
  model->setVisibleWeights(vw);
  model->setHiddenWeights(hw);
  model->setConditionalWeights(cw);
  model->setVisibleMean(getVisibleMean());
  model->setVisibleStd(getVisibleStd());
  model->setIsGaussian(getIsGaussian());

  return model;
}

}

}

