#Copyright (C) 2010 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

#
#see the bottom of file for code that instantiates and trains a model
#

import pylab
import numpy
import factored_gbm


def dispims(M, height, width, border=0, bordercolor=0.0, **kwargs):
    """ Display the columns of matrix M in a montage. """
    numimages = M.shape[1]
    n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    im = bordercolor*\
         numpy.ones(((height+border)*n1+border,(width+border)*n0+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[j*(height+border)+border:(j+1)*(height+border)+border,\
                   i*(width+border)+border:(i+1)*(width+border)+border] = \
                numpy.vstack((\
                  numpy.hstack((numpy.reshape(M[:,i*n1+j],(width,height)).T,\
                         bordercolor*numpy.ones((height,border),dtype=float))),\
                  bordercolor*numpy.ones((border,width+border),dtype=float)\
                  ))
    pylab.imshow(im.T,cmap=pylab.cm.gray,interpolation='nearest', **kwargs)
    pylab.show()


#instantiate the model
model = factored_gbm.FactoredGbmBinBin(numin=13*13, numout=13*13, nummap=100, numfactors=200, stepsize=0.01)

#fetch the data
inputimages = numpy.loadtxt('inputimages.txt').T
outputimages = numpy.loadtxt('outputimages.txt').T
numcases = inputimages.shape[1]

#train the model:
batchsize = 100
numbatches = numcases/batchsize
for epoch in range(200):
    print 'epoch ', epoch
    for batch in range(numbatches):
        model.train((inputimages[:, batch*batchsize:(batch+1)*batchsize], 
                     outputimages[:, batch*batchsize:(batch+1)*batchsize]), 
                     0.0, 1)


#visualize the filters:
pylab.clf()
pylab.subplot(1,2,1)
dispims(model.wxf, 13, 13, 2)
pylab.axis('off')
pylab.subplot(1,2,2)
dispims(model.wyf, 13, 13, 2)
pylab.axis('off')


