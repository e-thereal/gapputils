#Copyright (C) 2009-2010 Roland Memisevic
#
#This software and any documentation and/or information supplied with 
#it is distributed on an as is basis. Roland Memisevic makes no warranties, 
#express or implied, including but not limited to implied warranties 
#of merchantability and fitness for a particular purpose, regarding 
#the documentation, functions or performance of the software, 
#documentation and/or information. 
#

#This file contains classes that implement factored higher order 
#Boltzmann machines.  
#See:
#Roland Memisevic, Geoffrey Hinton (2010): 
#Learning to Represent Spatial Transformations with Factored Higher-Order Boltzmann Machines (Neural Computation). 



from numpy import zeros, ones, newaxis, array, asarray, double, dot, \
                  concatenate, log, exp, sum, reshape, isnan
import numpy.random

class FactoredGbm(object):
    """Factored gated Boltzmann machine model (base class).

       Gated Boltzmann machine whose weight-tensor is factorized as
       W_ijk=w_ij*w_jk*wik

       See:
       Roland Memisevic, Geoffrey Hinton (2010): 
       Learning to Represent Spatial Transformations with Factored Higher-Order Boltzmann Machines (Neural Computation). 
    """

    def __init__(self,numin,numout,nummap,numfactors,\
                 sparsitygain=0.0,targethidprobs=0.2,
                 cditerations=1,
                 meanfield_output=False,
                 premap=None,postmap=None,
                 optlevel=0,
                 momentum=0.9,
                 stepsize=0.001,
                 verbose=True):
        self.optlevel = optlevel
        self.numfactors = numfactors
        self.numin   = numin
        self.numout  = numout
        self.nummap  = nummap
        self.premap  = premap
        self.postmap = postmap
        if premap is not None:
            self.numin  = self.premap[1].shape[0]
            self.numout = self.postmap[0].shape[1]
        self.sparsitygain = sparsitygain
        self.targethidprobs = targethidprobs
        self.cditerations = cditerations
        self.meanfield_output = meanfield_output
        self.params  = 0.01*numpy.random.randn(self.numin*self.numfactors+\
                                        self.numout*self.numfactors+\
                                        self.nummap*self.numfactors+\
                                     self.numout+\
                                     self.nummap)
        self.numparams = self.params.shape[0]
        self.verbose = verbose
        self.numthreewayparams = numfactors*(numin+numout+nummap)
        self.wxf = self.params[:numin*numfactors].reshape((numin,numfactors))
        self.wyf = self.params[numin*numfactors:\
                           numin*numfactors+numout*numfactors].\
                           reshape((numout,numfactors))
        self.whf = self.params[numin*numfactors+numout*numfactors:
                           numin*numfactors+numout*numfactors+
                           nummap*numfactors].reshape((nummap,numfactors))
        self.wy = self.params[self.numthreewayparams:\
                           self.numthreewayparams+numout].\
                           reshape((numout,1))
        self.wh = self.params[self.numthreewayparams+numout:\
                           self.numthreewayparams+numout+nummap].\
                           reshape((nummap,1))
        self.wh*=0.0
        self.wy*=0.0
        self.prods_wxf = zeros(self.wxf.shape, 'double')
        self.prods_wyf = zeros(self.wyf.shape, 'double')
        self.prods_whf = zeros(self.whf.shape, 'double')
        self.momentum = momentum
        self.stepsize = stepsize
        self.inc = zeros(self.params.shape, 'double')

    def train(self, *args):
        numsteps = args[-1] 
        for step in range(numsteps):
            self.inc[:] = self.momentum*self.inc - self.stepsize * self.grad(*args[:-1])
            if isnan(sum(self.inc)): 
                print 'nan!'
                self.inc = numpy.zeros(self.inc.shape, 'double')
            self.params += self.inc

    def updateparams(self, newparams):
        self.params[:] = newparams.copy()

    def grad(self, data, weightcost):
        posgrad = self.energy_grad(self.posdata(data))
        if self.sparsitygain > 0.0:
            sparsitygrad = self.sparsitygrad(self.hids, data)
        else:
            sparsitygrad = 0.0
        neggrad = self.energy_grad(self.negdata(data))
        grad = -posgrad + neggrad
        grad += sparsitygrad
        weightcostgrad_x = weightcost*self.wxf.flatten()
        weightcostgrad_y = weightcost*self.wyf.flatten()
        weightcostgrad_h = weightcost*self.whf.flatten()
        weightcostgrad = concatenate((weightcostgrad_x,
                                      weightcostgrad_y,
                                      weightcostgrad_h))
        grad[:self.numthreewayparams] += weightcostgrad
        #grad[self.numthreewayparams:] *= 0.1 #smaller learningrate for biases
        return grad

    def energy_grad(self, data):
        inputs, hidprobs, outputs = data
        numcases = inputs.shape[1] 
        self.prods_wxf *= 0.0
        self.prods_wyf *= 0.0
        self.prods_whf *= 0.0
        factors_x = dot(inputs.T,self.wxf)
        factors_y = dot(outputs.T,self.wyf)
        factors_h = dot(hidprobs.T,self.whf)
        if self.optlevel > 0:
            if self.xy or self.yh or self.xh:
                assert False, "two-way biases not implemented for optlevel>0"
            from scipy import weave
            code = r"""
            for (int c=0; c<numcases; c++){
              for (int f=0; f<numfactors; f++){
                for (int i=0; i<numin; i++){
                  prods_wxf(i,f)+=inputs(i,c)*factors_y(c,f)*factors_h(c,f);
                }
                for (int j=0; j<numout; j++){
                  prods_wyf(j,f)+=outputs(j,c)*factors_x(c,f)*factors_h(c,f);
                }
                for (int k=0; k<nummap; k++){
                 prods_whf(k,f)+=hidprobs(k,c)*factors_x(c,f)*factors_y(c,f);
                }
              }
            }
            """
            global_dict = {'prods_wxf':self.prods_wxf,
                           'prods_wyf':self.prods_wyf,
                           'prods_whf':self.prods_whf,
                           'factors_x':factors_x,
                           'factors_y':factors_y,
                           'factors_h':factors_h,
                           'numin':self.numin,
                           'numout':self.numout,
                           'nummap':self.nummap,
                           'numfactors':self.numfactors,
                           'numcases':numcases,
                           'inputs':inputs,
                           'outputs':outputs,
                           'hidprobs':hidprobs,
                          }
            weave.inline(code,global_dict.keys(),global_dict=global_dict,
                                        type_converters=weave.converters.blitz)
        else:
            self.prods_wxf += sum(factors_y[:,newaxis,:]
                                     *factors_h[:,newaxis,:]
                                     *inputs.T[:,:,newaxis], 0)
            self.prods_wyf += sum(factors_x[:,newaxis,:]
                                     *factors_h[:,newaxis,:]
                                     *outputs.T[:,:,newaxis], 0)
            self.prods_whf += sum(factors_x[:,newaxis,:]
                                     *factors_y[:,newaxis,:]
                                     *hidprobs.T[:,:,newaxis], 0)
        visact = sum(outputs,1).flatten()
        hidact = sum(hidprobs,1).flatten()
        grad_wxf = self.prods_wxf.flatten()/double(numcases)
        grad_wyf = self.prods_wyf.flatten()/double(numcases)
        grad_whf = self.prods_whf.flatten()/double(numcases)
        grady = reshape(visact/double(numcases), self.numout) 
        gradh = reshape(hidact/double(numcases), self.nummap)
        return concatenate((grad_wxf,\
                            grad_wyf,\
                            grad_whf,\
                            grady,\
                            gradh\
                           ))

    def sparsitygrad(self, hidprobs, data):
        inputs, outputs = data
        numcases = double(hidprobs.shape[1])
        factors_x = dot(inputs.T,self.wxf)
        factors_y = dot(outputs.T,self.wyf)
        spxf = sum(sum(
                (hidprobs.T[:,:,newaxis,newaxis].transpose(0,2,1,3)-
                    self.targethidprobs)\
                * self.whf[newaxis,newaxis,:,:] \
                * factors_y[:,:,newaxis,newaxis].transpose(0,2,3,1) \
                * inputs[:,:,newaxis,newaxis].transpose(1,0,2,3) \
              ,2),0)/numcases
        spyf = sum(sum(
                 (hidprobs.T[:,:,newaxis,newaxis]\
                                .transpose(0,2,1,3)-self.targethidprobs) \
                 * self.whf[newaxis,newaxis,:,:] \
                 * factors_x[:,:,newaxis,newaxis].transpose(0,2,3,1) \
                 * outputs[:,:,newaxis,newaxis].transpose(1,0,2,3) \
               ,2),0)/numcases
        sphf = sum((hidprobs.T[:,:,newaxis]-self.targethidprobs) \
                  * factors_x[:,newaxis,:] \
                  * factors_y[:,newaxis,:]\
                  ,0)/numcases
        sph = (sum(hidprobs-self.targethidprobs,1)/numcases).flatten()
        spgrad_wxf = (self.sparsitygain*spxf).flatten()
        spgrad_wyf = (self.sparsitygain*spyf).flatten()
        spgrad_whf = (self.sparsitygain*sphf).flatten()
        spgrady = zeros(self.numout,  'double')
        spgradh = self.sparsitygain * sph
        return concatenate((spgrad_wxf,\
                            spgrad_wyf,\
                            spgrad_whf,\
                            spgrady,\
                            spgradh\
                           ))

    def factors_x(self, inputs):
        return dot(inputs.T, self.wxf)

    def factors_y(self, outputs):
        return dot(outputs.T, self.wyf)

    def factors_h(self, hiddens):
        return dot(hiddens.T, self.whf)

    def posdata(self,data):
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        if self.premap is not None:
            inp = dot(self.premap[1], inp+self.premap[0])
            out = dot(self.premap[1], out+self.premap[0])
        self.hids = self.hidprobs(out, inp)
        return inp, self.hids, out

    def negdata(self, data):
        if len(data[0].shape) < 2: #got rank-1 arrays?
            inp = data[0].reshape((data[0].shape[0],1))
            out = data[1].reshape((data[1].shape[0],1))
        else:
            inp = data[0]
            out = data[1]
        numcases = inp.shape[1]
        out_original = out
        if self.premap is not None:
            inp = dot(self.premap[1], inp+self.premap[0])
            out = dot(self.premap[1], out+self.premap[0])
        if self.cditerations==1:
            hidstates = self.sample_hid(self.hids)
            negoutput = self.outprobs(hidstates, inp)
            if not self.meanfield_output:
                negoutput = self.sample_obs(negoutput)
            neghidprobs = self.hidprobs(negoutput, inp)
        else:
            for c in range(self.cditerations):
                hidstates = self.sample_hid(self.hids)
                negoutput = self.outprobs(hidstates, inp)
                datastates= self.sample_obs(negoutput)
                self.hids = self.hidprobs(datastates,inp)
            neghidprobs = self.hidprobs(datastates, inp)
        if self.verbose:
            if self.postmap is not None:
                print "av. squared reconstrution err: %f" % \
                     (sum(sum((\
                 out_original-(dot(self.postmap[0],negoutput)+self.postmap[1])\
                     )**2))/double(numcases))
            else:
                print "av. squared reconstrution err: %f" % \
                      (sum(sum(asarray(out-negoutput)**2))/double(numcases))
        return inp, neghidprobs, negoutput

    def factors_x(self, inputs):
        return dot(inputs.T, self.wxf)

    def factors_y(self, outputs):
        return dot(outputs.T, self.wyf)

    def factors_h(self, hiddens):
        return dot(hiddens.T, self.whf)

    def recons(self, x, y):
        return self.outprobs(self.hidprobs(y, x), x)

    def rdist(self, x1, x2):
        """ 'Reconstruction distance.' """
        if len(x1.shape)<2: x1 = x1[:,newaxis]
        if len(x2.shape)<2: x2 = x2[:,newaxis]
        dists = zeros((x1.shape[1],x2.shape[1]), 'double')
        for j in range(x2.shape[1]):
            dists[:,j] = \
                ((self.recons(x1, x2[:,j][:,newaxis].repeat(x1.shape[1],1))-
                                                x2[:,j][:,newaxis])**2).sum(0)
        return dists


class FactoredGbmBinBin(FactoredGbm):

    def hidprobs(self, outputs, inputs):
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        numcases = outputs.shape[1]
        if len(inputs.shape)<2: #got rank-1 array?
            inputs=inputs[:,newaxis]
        factors_x = self.factors_x(inputs)
        factors_y = self.factors_y(outputs)
        if self.optlevel > 0:
            from scipy import weave
            result = zeros((self.nummap,numcases), 'double')
            code = r"""
              #include <math.h>
              for(int c=0;c<numcases;c++){
                for(int k=0;k<nummap;k++){
                  for(int f=0;f<numfactors;f++){
                    result(k,c) -= factors_x(c,f)*factors_y(c,f)*whf(k,f);
                  }
                }
              }
              for(int c=0;c<numcases;c++){
                for(int k=0;k<nummap;k++){
                  result(k,c) -= wh(k);
                  result(k,c) = 1.0/(1.0+exp(result(k,c)));
                }
              }
            """
            global_dict = {'factors_x':factors_x,
                           'factors_y':factors_y,\
                           'whf':self.whf,\
                           'wh':self.wh,\
                           'nummap':self.nummap,\
                           'numfactors':self.numfactors,\
                           'numcases':numcases,\
                           'result':result}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,\
                         type_converters=weave.converters.blitz)
            return result
        else:
            return 1.0/(1.0+exp(-(
                        sum(factors_x[:,newaxis,:]*factors_y[:,newaxis,:]*
                                   self.whf[newaxis,:,:],2)
                            + self.wh.T\
                 ))).T

    def outprobs(self, hiddens, inputs):
        if len(hiddens.shape)<2: #got rank-1 array?
            hiddens=hiddens[:,newaxis]
        numcases = hiddens.shape[1]
        if len(inputs.shape)<2: #got rank-1 array?
            inputs=inputs[:,newaxis]
        factors_x = self.factors_x(inputs)
        factors_h = self.factors_h(hiddens)
        return 1.0/(1.0+exp(-(\
               sum(factors_x[:,newaxis,:]*factors_h[:,newaxis,:]*
                                    self.wyf[newaxis,:,:],2)
               + self.wy.T\
               ))).T

    def sample_hid(self,hidprobs):
        return (hidprobs > numpy.random.rand(hidprobs.shape[0],\
                                             hidprobs.shape[1])).astype(float)

    def sample_obs(self,outprobs):
        return (outprobs > numpy.random.rand(outprobs.shape[0],\
                                             outprobs.shape[1])).astype(float)

class FactoredGbmBinGauss(FactoredGbm):
    def __init__(self,numin,numout,nummap,numfactors,\
                 sparsitygain=0.0,targethidprobs=0.1,nu=1.0,cditerations=1,\
                 premap=None,postmap=None,\
                 xy = False, xh = False, yh = True,\
                 momentum=0.9,
                 stepsize=0.01,
                 verbose=0):
        FactoredGbm.__init__(self,numin,numout,nummap,numfactors,\
                   sparsitygain,targethidprobs,cditerations=cditerations,\
                   premap=premap,postmap=postmap,\
                   xy=xy,xh=xh,yh=yh,momentum=momentum,
                   stepsize=stepsize,verbose=verbose)
        self.meanfield_output = False
        self.nu = nu

    def hidprobs(self, outputs, inputs):
        if len(outputs.shape)<2: #got rank-1 array?
            outputs=outputs[:,newaxis]
        numcases = outputs.shape[1]
        if len(inputs.shape)<2: #got rank-1 array?
            inputs=inputs[:,newaxis]
        factors_x = self.factors_x(inputs)
        factors_y = self.factors_y(outputs)
        if self.optlevel > 0:
            from scipy import weave
            result = zeros((self.nummap,numcases), 'double')
            code = r"""
              #include <math.h>
              #include <Python.h>
              Py_BEGIN_ALLOW_THREADS
              for(int c=0;c<numcases;c++){
                for(int k=0;k<nummap;k++){
                  for(int f=0;f<numfactors;f++){
                    result(k,c) -= factors_x(c,f)*factors_y(c,f)*whf(k,f);
                  }
                }
              }
              for(int c=0;c<numcases;c++){
                for(int k=0;k<nummap;k++){
                  result(k,c) -= wh(k);
                  result(k,c) = 1.0/(1.0+exp(result(k,c)));
                }
              }
              Py_END_ALLOW_THREADS
            """
            global_dict = {'factors_x':factors_x,
                           'factors_y':factors_y,\
                           'whf':self.whf,\
                           'wh':self.wh,\
                           'nummap':self.nummap,\
                           'numfactors':self.numfactors,\
                           'numcases':numcases,\
                           'result':result}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,\
                         type_converters=weave.converters.blitz)
            return result
        else:
            return 1.0/(1.0+exp(-(\
                 sum(factors_x[:,newaxis,:]*factors_y[:,newaxis,:]*
                                   self.whf[newaxis,:,:],2)
                 + self.wh.T\
                 ))).T

    def outprobs(self, hiddens, inputs):
        if len(hiddens.shape)<2: #got rank-1 array?
            hiddens=hiddens[:,newaxis]
        numcases = hiddens.shape[1]
        if len(inputs.shape)<2: #got rank-1 array?
            inputs=inputs[:,newaxis]
        factors_x = self.factors_x(inputs)
        factors_h = self.factors_h(hiddens)
        if self.optlevel > 0:
            from scipy import weave
            result = zeros((self.numout,numcases), 'double')
            code = r"""
              #include <math.h>
              #include <Python.h>
              Py_BEGIN_ALLOW_THREADS
              for(int c=0;c<numcases;c++){
                for(int j=0;j<numout;j++){
                  for(int f=0;f<numfactors;f++){
                    result(j,c) += factors_x(c,f)*factors_h(c,f)*wyf(j,f);
                  }
                }
              }
              for(int c=0;c<numcases;c++){
                for(int j=0;j<numout;j++){
                  result(j,c) += wy(j);
                  result(j,c) /= nu;
                  result(j,c) *= nu2;
                }
              }
              Py_END_ALLOW_THREADS
            """
            global_dict = {'factors_x':factors_x,
                           'factors_h':factors_h,\
                           'wyf':self.wyf,\
                           'wy':self.wy,\
                           'numout':self.numout,\
                           'nu':self.nu,\
                           'nu2':self.nu**2,\
                           'numfactors':self.numfactors,\
                           'numcases':numcases,\
                           'result':result}
            weave.inline(code,global_dict.keys(),global_dict=global_dict,\
                         type_converters=weave.converters.blitz)
            return result
        else:
            return ((sum(factors_x[:,newaxis,:]*factors_h[:,newaxis,:]*
                                    self.wyf[newaxis,:,:],2)
                   + self.wy.T)/self.nu).T * self.nu**2

    def sample_hid(self, hidprobs):
        return (hidprobs > numpy.random.rand(*hidprobs.shape)).astype(float)

    def sample_obs(self, outprobs):
        return (outprobs + numpy.random.randn(*outprobs.shape))*self.nu

