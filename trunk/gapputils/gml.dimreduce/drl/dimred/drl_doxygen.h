/***************************************************************************
                                drl_doxygen.h
                             -------------------
                               W. Michael Brown

  Documentation for Dr.L. Library

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Wed Aug 10 2005
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

/** 

\mainpage Dimensionality Reduction Library (Dr.L.)
\section intro Introduction
Dr. L. is a C++ library for dimensionality reduction. In the context of
this library, dimensionality reduction is considered to consist of:
<ol>
-#
-# Estimation of the intrinsic dimensionality using sampled data
-# Finding maps that reduce the dimensionality of data 
   (forward map) or increase the dimensionality of data (reverse map).
   The "find" is performed by optimizing some metric based on the sampled 
   data. In the ideal case, applying the forward map followed by the 
   reverse map should regenerate the original data points.
-# Mapping arbitrary coordinates to high and low dimensionalities
</ol>

The library is intended to provide a consistent interface to multiple
dimensionality reduction algorithms with an efficient C++ interface that
runs efficiently on multicore architectures. A few routines have been
optimized with an option for GPU acceleration or distributed computation.

Currently, the library offers intrinsic dimensionality estimation using:
<ol>
-#
-# point-PCA
-# reconstruction error
-# residual variance. 
</ol>

The following dimensionality reduction methods have been implemented.
<ol>
-#
-# Principal Component Analysis
-# Multidimensional Scaling
-# Locally Linear Embedding
-# IsoMap
-# Autoencoder Neural Networks
</ol>

An executable is also supplied that can be built to allow for
command-line access to the library routines. A description of an application
of the library for molecular structure analysis has been published
\ref ref "[Brown, 2009]".

\n
\n
\section dimred Dimensionality Reduction

In general, dimensionality reduction algorithms provide a method
for taking a set of samples \f$\{\mathbf{x}_1, \dots, \mathbf{x}_n \} \subset
\mathbb{R}^D\f$ and calculating a corresponding low-dimensional representation
\f$\{\mathbf{y}_1, \dots, \mathbf{y}_n \} \subset
\mathbb{R}^d\f$. Because dimensionality reduction is often
used for visualization, some algorithms do not
generate an explicit map from the high dimensional
coordinates to the low dimensional representation. 
For many applications, however, it is desirable to have 
an explicit forward map, \f$\Phi(\mathbf{x}) : \mathbb{R}^D \rightarrow 
\mathbb{R}^d\f$, that gives the low-dimensional representation 
of an arbitrary point \f$\mathbf{x}\f$ and an explicit reverse map 
\f$\phi(\mathbf{y}) : \mathbb{R}^d \rightarrow \mathbb{R}^D\f$ 
that gives the high-dimensional
represenation of an arbitrary point \f$\mathbf{y}\f$. This allows for mapping new
samples that were not available at the time of the initial reduction and
also provides a common metric for comparison of algorithms. 
Therefore, for the purposes of this work, we consider dimensionality
reduction as the problem of generating \f$\Phi\f$ and \f$\phi\f$ from a training
set of \f$n\f$ samples, \f$X_{D \times n} = (\mathbf{x}_1, \dots, \mathbf{x}_n )\f$.
Because some methods do not generate explicit maps, we describe an 
approach for generating maps from a dimensionality reduction below.

The performance of each algorithm can be evaluated using the reconstruction
error. Ideally, a forward map of an arbitrary point followed by a 
reverse map will give the same point back.
Typically, the reconstruction error is given by 
\f$\|\mathbf{x}-\phi(\Phi(\mathbf{x}))\|\f$.

\n
\n
\section ns Namespace

Although hidden from doxygen, the Dr. L. library
resides in the \e yala namespace. You can access Dr. L. commands in your code
with the following lines:

\code
#include "dimred/ya_dimred.h"
using namespace yala;
\endcode

\n
\n
\section vecmat Matrix and Vector Storage in Dr. L.

Dr. L. uses a custom linear algebra library for computations. To incorporate
the library with your code, wrappers can be used for column-major or row-major
matrices stored in traditional continuous memory arrays, STL vectors, or
standard library valarrays. The wrapper used will depend on whether each
datapoint is stored in a row or a column and whether or not the matrix is
column major. 

\code
#include "dimred/ya_dimred.h"
using namespace yala;

int main(int argc, char **argv) {
  int rows=4;
  int cols=3;
  int numel=rows*cols;
  double input[numel];
  
  // Fill the matrix....
  
  // If each row is a datapoint (each column is a dimension)
  // --If the matrix is column major
  YA_WRAP(double) input_w(input,rows,cols);
  // --If the matrix is row major
  YA_WRAP_RM(double) input_w(input,rows,cols);
  
  // If each column is a datapoint (each row is a dimension)
  // --If the matrix is column major
  YA_WRAP_RM(double) input_w(input,cols,rows);
  // --If the matrix is row major
  YA_WRAP(double) input_w(input,cols,rows);

  // Use input_w with Dr. L. routines...
  // Same procedure can be used for output matrices.
  
  // Use input as usual...
  
  return 0;
}
\endcode 

For single precision, double is replaced with float in the wrappers above, etc.
For STL vectors:

\code
  std::vector<double> input;
  
  ...
  
  
  YA_VWRAP(double) input_w(input.begin(),rows,cols);
  
  // or
  
  YA_VWRAP_RM(double) input_w(input.begin(),rows,cols);
  
\endcode

For valarray:

\code
  std::valarray<double> input;
  
  ...
  
  YA_WRAP(double) input_w(&input[0],rows,cols);
  
  // or
  
  YA_WRAP_RM(double) input_w(&input[0],rows,cols);
\endcode

\n
\n
\section general General Dimensionality Reduction Routines

Several general dimensionality reduction routines are available through
the pure virtual class YADimReduce. The base class can also be used for
run-time selection of dimensionality routines:

\code
  YADimReduce<double> *redmet=NULL;
  
  if (method_str=="PCA")
    redmet=new YAPCAReduce<double>;
  else if (method_str=="MDS")
    redmet=new YAMDSReduce<double>;
  else if (method_str=="LLE")
    redmet=new YALLEReduce<double>;
  else if (method_str=="ISOMAP")
    redmet=new YAIsoReduce<double>;
  else if (method_str=="AE")
    redmet=new YAATEReduce<double>;
\endcode

The following sample code describes some of the routines available through the 
base class:

\code
  // Store the high dimensionality and low dimensionality for maps somewhere
  int high_dim=5;
  int low_dim=3;
  
  // For methods that utilize eigen decomposition, options are set through
  // the EigenOptions class.
  EigenOptions eigopts;
  
  // Set the degree of output to the console
  // -- 0 is not output, 1 outputs a progress meter, 2 also outputs nbor stats
  redmet->verbose(2);

  // The find routine uses a set of input samples to obtain forward and reverse
  // maps. Additionally, this routine will store the low-dimensional 
  // representation for a set of samples in an output matrix. 
  redmet->find_t(input, output, low_dim, eigopts);
  
  // Map in the forward direction to reduce the dimensionality of a matrix
  redmet->forward_t(input_mat, output_mat);
  
  // Map in the reverse direction to increase the dimensionality of a matrix
  redmet->reverse_t(input_mat, output_mat);
  
  // Calculate RMSD representing the reconstruction error from the input matrix
  double err = redmet->reconstruct_error(input);  
    
  // Save dimensionality reduction maps to the given file name
  int error_flag=redmet->save_map("example_maps.dat");
  
  // Load dimensionality reduction maps from the given file name
  int error_flag=redmet->load_map("example_maps.dat");
\endcode

For a list of additional routines, click on YADimReduce.

\n
\n
\section nbors Neighbor Reconstruction Mapping

LLE and Isomap produce a low-dimensional embedding 
\f$Y_{d \times n} = \{\mathbf{y}_1, \dots, \mathbf{y}_n \} \subset
\mathbb{R}^d\f$ from the samples in \f$X\f$ without generating an explicit
map. Here, we have considered dimensionality reduction as a problem
of finding the maps \f$\Phi\f$ and \f$\phi\f$ from training data. For LLE and
Isomap, we accomplish this with the maps \f$\Phi_{NRM}(X,Y,\mathbf{x})\f$ and
\f$\phi_{NRM}(X,Y,\mathbf{y})\f$ that allow for dimensionality reduction to
be performed on future samples based on the initial embedding of
training data. A natural choice for these maps 
is some method that retains the positioning of a
sample relative to its neighbors in the training set.
Because LLE and Isomap assume that a sample and its neighbors are locally
linear, we can perform the mapping using a linear combination of
a sample's \f$k\f$ neighbors:

\f[
\Phi_{NRM}(X,Y,\mathbf{x})=\displaystyle\sum_{i=1}^k w_i\mathbf{y}_i
\f]
and
\f[
\phi_{NRM}(X,Y,\mathbf{y})=\displaystyle\sum_{i=1}^k w_i\mathbf{x}_i.
\f]

That is, the training set neighbors for an arbtrary point \f$\mathbf{x}\f$ or 
\f$\mathbf{y}\f$ can be identified in the input dimensionality and
used to determine the sample mapping based on their positions
(\f$\mathbf{x}_i\f$ or \f$\mathbf{y}_i\f$) in the desired dimensionality. 
The question is
how to choose the weights \f$w_i\f$. The equations bear a strong
resemblence to the reconstruction approach
used in LLE and it has been
suggested that this same approach can used to map new 
samples \ref ref "[Saul, 2003]". In this case, \f$w_i\f$ are determined
in a least-squares optimization with a closed form solution.
There are issues in implementing this approach, however. For the
case when the number of neighbors \f$k\f$ is greater than the intrinsic
dimensionality of the manifold, the solution for \f$w_i\f$ is not unique.
Because it can be desirable that \f$k\f$ is variable and because the
intrinsic dimensionality is not necessarily known \e a \e priori, it is
not straightforward to decide when the problem must be conditioned 
to provide a unique solution. Therefore, although this approach is an
option in Dr. L., a simpler alternative is the default. In this case,
\f$w_i\f$ is chosen to be the inverse Euclidean distance between
the sample and the neighbor \f$i\f$. This approach allows for an
arbitrarily high number of neighbors, however, will clearly fail in
the case when a sample is outside the convex hull of its neighbors
(due to the constraint that \f$w_i\f$ is positive).

When forward or reverse mapping is performed using a method that does not
obtain explicit maps (e.g. LLE and IsoMap), neighbor reconstruction mapping (NRM)
will be performed. In this case, the user has the choice to use LLE weights
or distance weights (as described above). Additionally, either k-nearest neighbors
or epsilon neighbors can be used:

\code
  // Set the weights for reconstruction based on distances
  // -- Alternatively, 1 can be specified to use LLE least-squares weighting.
  redmet->neighbor_weight_mode(0);
  
  // If we are using k-nearest neighbors, specify k
  int k=10;
  redmet->neighbors(k);
  
  // If we are using a ball, specify epsilon
  double eps=0.2;
  redmet->epsilon(eps);
  
  // Specify that we are using k-nearest neighbors
  // -- Alternatively, we could use 1 to specify epsilon neighbors
  redmet->neighbor_mode(0);

  // Perform a forward map
  redmet->forward_t(input,output);
\endcode

For details, see YADimReduce.

\n
\n
\section pca Principal Component Analysis
PCA is a linear dimensionality reduction approach that has been widely
applied to problems in almost 
every field of experimental science. The goal of PCA is to find a coordinate 
representation for data where the most variance is captured in the 
least number of coordinates. This representation can be found by performing 
an eigenvalue decomposition (or singular value decomposition)
such that the resulting eigenvectors/singular vectors
provide an orthonormal basis for the data while the eigenvalues/singular
values provide 
information on the importance of each basis vector. Given the training
set \f$X\f$, a row-centered matrix is calculated as \f$\widetilde{X}_{D \times n} =
(\widetilde{\mathbf{x}}_1, \dots, \widetilde{\mathbf{x}}_n)\f$, where
\f$\widetilde{\mathbf{x}}_i = \mathbf{x}_i - \mathbf{m}\f$ and 
\f$\mathbf{m}_{D \times 1}\f$ gives the row means. Eigen decomposition of the
training set covariance matrix, \f$\frac{1}{n} \widetilde{X} \widetilde{X}^T\f$,
is performed to give \f$U P U^T\f$. The forward map is then given by
\f$\Phi_{PCA}(\mathbf{x})=\widehat{U}^T (\mathbf{x}-\mathbf{m})\f$, where
\f$\widehat{U}_{d \times n}\f$ is the matrix composed of
the first \f$d\f$ columns of \f$U\f$ corresponding to the eigenvectors
with the largest eigenvalues. The reverse map is calculated as
\f$\phi_{PCA}(\mathbf{y})=\widehat{U} \mathbf{y} + \mathbf{m}.\f$ The
reconstruction error for PCA will be zero for \f$d>=D-z\f$, where \f$z\f$ is
the number of non-zero eigenvalues in \f$P\f$. For details on the PCA routines,
see YAPCAReduce.

\n
\n
\section lle Locally Linear Embedding

Locally Linear Embedding (LLE) is a nonlinear dimensionality reduction
method. LLE is performed by first solving
for the location of each sample \f$\mathbf{x}_i\f$ in terms 
of its neighbors. For each sample, the neighbors are determined as
all samples within a ball of specified radius centered on the sample
or as the \f$k\f$ nearest neighbors. A weight matrix, \f$W\f$, is obtained
by determining the weights in a linear combination of neighbors
that best reconstruct each sample,

\f{eqnarray}
\text{min}_W & E(W) = \sum_i \| \mathbf{x}_i - \sum_j w_{ij} \mathbf{x}_j \|^2 \smallskip \\
\text{subject to} & \left\{ \begin{array}{l}
w_{ij} = 0 \text{ if } \mathbf{x}_i \text{ not neighbor }
\mathbf{x}_j\\
\sum_j w_{ij} = 1 \text{ for every } i,
\end{array} \right.
\f}

where \f$W = (w_{ij})\f$.  This problem has a closed
form solution and assures not only that each approximation
\f$\mathbf{x}_i \approx \sum_j w_{ij} \mathbf{x}_j\f$ lies in the subspace
spanned by the \f$k\f$ neighbors of \f$\mathbf{x}_i\f$, but also that the
solution \f$W\f$ is invariant to translation, rotation, and rescaling.
These properties allow, by design, calculation of a linear mapping that
is also invariant to translation, rotation, and rescaling. This mapping
from the \f$n\f$ data samples \f$\mathbf{x}_i\f$ to the low dimensional embedding
\f$\mathbf{y}_i\f$ is performed by minimizing the embedding cost function,

\f[
\Gamma=\displaystyle\sum_{i=1}^n\|\mathbf{y}_i-\sum_{j=1}^n 
w_{ij}\mathbf{y}_j\|^2.
\f]

In this case, the weights \f$w_{ij}\f$ are fixed and the 
low-dimensional coordinates are optimized.
This is a quadratic minimization problem with a unique global minimum.
It can be solved as a sparse \f$n \times n\f$ eigen value problem where
the bottom \f$d\f$ non-zero eigenvectors provide the embedding
(the bottom eigenvalue is zero). From 
this, it can be seen that LLE assumes that
a sample and its neighbors can be treated in a linear fashion. Global
structure is maintained due to the overlap of neighbors in each
local patch in the embedding cost function. A detailed description
of LLE can be found in \ref ref "[Roweis, 2000]" and 
\ref ref "[Saul, 2003]".

Because the low-dimensional representation is optimized directly,
no explicit maps are generated. Here, we use \f$\Phi_{NRM}\f$ and
\f$\phi_{NRM}\f$ to perform mapping in terms of the initial LLE reduction
as described above. LLE is parameterized by the neighboring method
(kNN or epsilon) and any options for eigen decomposition. These should
be set before performing a find operation:

\code
  // Use default eigen decomposition options
  EigenOptions eigopts;

  // If we are using k-nearest neighbors, specify k
  int k=10;
  redmet->neighbors(k);
  
  // If we are using a ball, specify epsilon
  double eps=0.2;
  redmet->epsilon(eps);
  
  // Specify that we are using k-nearest neighbors
  // -- Alternatively, we could use 1 to specify epsilon neighbors
  redmet->neighbor_mode(0);

  // Set the dimensionality to reduce to
  int low_dim=2;
  
  // Perform the reduction
  redmet->find_t(input, output, low_dim, eigopts);
\endcode

See YALLEReduce for further details.

\n
\n
\section iso IsoMap

Isomap is a nonlinear dimensionality reduction algorithm, 
described in \ref ref "[Tenenbaum, 2000]". The first step in the 
Isomap algorithm is to impose a graph structure 
\f$G(V,E,W)\f$ on the input dataset \f$X\f$. Each sample \f$\mathbf{x}_i\in X\f$ is 
represented by a node \f$v_i \in V\f$ and two nodes \f$v_i\f$ and \f$v_j\f$ are connected 
by an edge \f$(v_i, v_j) \in E\f$ with weight \f$w_{ij} \in W\f$ if \f$\mathbf{x}_i\f$ 
is a a neighbor of \f$\mathbf{x}_j\f$. Neighbors are calculated in the same manner 
as perfromed in LLE. The weight of \f$w_{ij}\f$ is given by 
the Euclidean distance between \f$\mathbf{x}_i\f$ and \f$\mathbf{x}_j\f$. 
The second step in Isomap involves 
computation of the shortest paths between all nodes in \f$G\f$.
These distances are stored pairwise in a matrix \f$D_{G}\f$. The distance
matrix \f$D_{G}\f$ is intended to represent the distances between all
samples on the manifold - the geodesic distances. Because these
distances are Euclidean for each sample and its neighbors, Isomap
makes the same assumption of local linearity as LLE. Unlike LLE,
global distances between all neighbors are explicity calculated
with the graph approximation to geodesic distances. 

Because all
pairwaise distances are available, Multi-Dimensional Scaling (MDS) 
can be applied to \f$D_{G}\f$ to perform a low-dimensional embedding. 
MDS is a variant of PCA that starts with a distance matrix \f$D_G\f$, 
converts the distance matrix to an inner product matrix, and calculates the 
eigenvalue decomposition of the resulting matrix. 
For the case presented here, this is performed by
squaring each element in the distance matrix \f$D_{G}\f$, 
double-centering the resulting matrix, and performing the eigenvalue
decomposition to give \f$U P U^T\f$. The low-dimensional embedding is
then given by \f$Y=\widehat{U} \widehat{P}\f$, where
\f$\widehat{U}_{d \times n}\f$ is the matrix comprised by
the first \f$d\f$ columns of \f$U\f$ corresponding to the eigenvectors with
largest eigenvalues and \f$\widehat{P}_{d \times d}\f$ is the
diagonal matrix containing the square roots of the largest \f$d\f$ eigenvalues.

Like LLE, Isomap does not calculate explicit maps in order to perform
an embedding. Here, we use \f$\Phi_{NRM}\f$ and
\f$\phi_{NRM}\f$ to perform mapping in terms of the initial Isomap reduction
as described above. Also like LLE, IsoMap is parameterized by the neighboring method
(kNN or epsilon) and any options for eigen decomposition. These should
be set before performing a find operation:

\code
  // Use default eigen decomposition options
  EigenOptions eigopts;

  // If we are using k-nearest neighbors, specify k
  int k=10;
  redmet->neighbors(k);
  
  // If we are using a ball, specify epsilon
  double eps=0.2;
  redmet->epsilon(eps);
  
  // Specify that we are using k-nearest neighbors
  // -- Alternatively, we could use 1 to specify epsilon neighbors
  redmet->neighbor_mode(0);

  // Set the dimensionality to reduce to
  int low_dim=2;
  
  // Perform the reduction
  redmet->find_t(input, output, low_dim, eigopts);
\endcode

IsoMap relies on a graph constructed from each point and its neighbors and
there is no guarantee that a single connected graph will result from the
neighboring procedures. In the case where multiple connected components are
found, dimensionality reduction is performed separately on each component.
Future mappings using NRM based on this dimensionality reduction will map
points into their nearest component. Statistics on the number and sizes
of connected components using different neighboring criteria can be
obtained with the YAIsoReduce::component_stat() routines.

The runtime required for reduction with IsoMap can be reduced by using
\e landmark points. In this case, a subset of points from the sampled data
are chosen as landmark points that are used for computation of the distance
matrix and for performing future mapping using NRM. In this case, a vector
of indices for the landmarks is also supplied to the find routine (0 is first
index). [For wrappers, the vector can be a row vector or a column vector.]

\code
  YAIsoReduce redmet;
  
  ...
  
  redmet.findmap(input, output, low_dim, eigopts, landmarks);
\endcode

For details on these and other routines provided, see YAIsoReduce.

\n
\n
\section ate Autoencoder Neural Network

An \e autoencoder performs dimensionality reduction via a bottleneck
architecture neural network.  Autoencoders were originally introduced
sometime in the early 1990s, but they have not been
widely applied due to the extreme difficulty of the optimization
problem associated with training the resulting network. However, a method
was recently proposed for pre-training an autoencoder neural network using 
a Restricted Boltzmann Machine (RBM) in order to accelerate the optimization
process \ref ref "[Hinton, 2006]". This method was used to obtain impressive results 
on a very large benchmark dataset of hand written digits.

The autoencoder introduced in \ref ref "[Hinton, 2006]" consists of weighted sums
and compositions of the well-known function \f$\sigma(x) = 1/(1+\exp(x))\f$.  
These functions are separated into distinct layers, with interconnections
between functions in adjacent layers defining the network structure. 
At each layer in the network, inputs into the next layer consist of terms 
of the form \f$\sigma ( b_j + \sum_i v_i w_i )\f$, where \f$b_j\f$ represents a bias,
\f$w_i\f$ represents a weight, and \f$v_i\f$ represents an input from the previous 
network layer. The inputs to the first layer are taken to be the components 
of the original vectors in our dataset \f$X = \{\mathbf{x}_1, \dots, 
\mathbf{x}_n \}\f$. The
weights and biases are then optimized such that the mean reconstruction error
\f$1/n \sum_i \|\mathbf{x}_i - \phi_{AE}(\Phi_{AE}(\mathbf{x}_i))\|\f$ is 
minimized (where \f$\Phi_{AE}\f$ is the forward map and \f$\phi_{AE}\f$ is the reverse
map given by the network).

To provide an illustrative example, suppose we have a dataset \f$X\f$ with native 
dimension 784, for which we want to construct a 2-dimensional embedding. We 
first define a network structure such as 784--1000--500--250--2, where the 
integers in the sequence represent the number of \f$\sigma\f$ functions in each 
layer. When appropriately trained, this structure will perform a reduction
of 784-dimensional data to a 2-dimensional embedding. The justification for
the initial increase in dimension to 1000 is that because the \f$\sigma\f$ 
functions are inherently binary, we may experience a loss of information when
going from normalized data in \f$[0,1]\f$ to values in \f${0,1}\f$; the possible loss 
of information resulting from this process is potentially counter-balanced
by an initial increase in dimensionality. The encoding structure is then mirrored to
form a 2--250--500--1000--784 decoding network structure. The encoder and
decoder networks are then joined and training is performed on
the aggregate network.

As mentioned above, the optimization procedure for obtaining the
autoencoder weights proceeds in two steps.  In the first step, a RBM
is trained.  This training is performed for a user specified number of
iterations.  In the second step, the autoencoder weights are
fine-tuned using back-propagation (BP).  This step is also performed
for a user specified number of iterations.  In both cases a training
set is used for the optimization and a test set is used to avoid
overtraining.  The training set is also split into batches to avoid
overtraining, as well as to improve algorithm speed.  During each
iteration all of the batches are used in sequence.

The layers of the neural network and corresponding weights yield
an analytic expression for both the forward (\f$\Phi_{AE}\f$) and 
reverse (\f$\phi_{AE}\f$) maps that is optimized during training. This
allows for future mapping of arbitrary points.

When using the autoencoder, several additional commands must be executed
before the find routine in order to parameterize the random number generator
and the sizes for the layers. Additionally, many other parameters can
be tuned for autoencoder training. The most important parameters are
given in the example below. 

\code
  // Set up a random number generator and seed it
  MathRandom<MathMersenneTwister> rng;
  unsigned long seed=123456789;
  rng.seed(seed);
  
  // Get a dimensionality reduction object
  YAATEReduce redmet;
  // Low dimensionality
  int low_dim=4;
  
  // Tell redmet which random number generator to use.
  redmet.set_rng(&rng);
  // Give the layer sizes for the neural network
  int layers[4]={35 64 32 4};
  redmet.ae_layers(YA_WRAP(layers,1,4));
  
  // Fraction of the input matrix to be used for training
  double trac=0.8;
  redmet.train_split(tfrac);
  
  // Iterations of RBM and back-propagation
  redmet.rbm_iters(10);
  redmet.bp_iters(30);

  // Number of samples in RBM batches and BP batches
  redmet.rbm_size(10);
  redmet.bp_size(100);
  
  ...
  
\endcode
  
For additional routines, parameters, and test set specification,
see YAATEReduce.


\n
\n
\section intrinsic Intrinsic Dimensionality Estimation

Dimensionality reduction methods give an approach for obtaining a map 
\f$\Phi(\mathbf{x}) : \mathbb{R}^D \rightarrow \mathbb{R}^d\f$.
In addition to having a method to calculate the maps, we must
also determine an appropriate value for \f$d\f$.
One obvious choice is to determine some metric for quantifying
the success of dimensionality reduction and evaluate the
reduction performance at different embedding
dimensionalities. For PCA and MDS (IsoMap), this
metric can be the residual variance. The eigenvalues obtained
in these approaches give the variance in each dimension and
therefore the sum of the \f$d+1\f$ to \f$D\f$ eigenvalues is a measure
of the variance that is not accounted for in the reduction.
When this value is near zero, little is gained from adding a 
dimension. Although LLE also solves an eigenproblem, the
eigenvalues obtained have been shown to be unreliable in
determining \f$d\f$ \ref ref "[Saul, 2003]".

To obtain the eigenvalues from PCA, MDS, IsoMap, and LLE in Dr. L.:

\code
  int high_dim=20;
  int low_dim=3;
  
  ...
  
  redmet.find_t(input, output, low_dim, eigopts);
  
  double eigen_values[high_dim];
  YA_WRAP(double) eig_w(eigen_values,1,20);
  eig_w=redmet.eigenvalues();
\endcode

An alternative metric utilized in Isomap \ref ref "[Tenenbaum, 2009]"
is a geodesic distance correlation residual given by \f$1-R^2(D_G,D_Y)\f$, 
where \f$R^2(D_G,D_Y)\f$ is the correlation coefficient between geodesic
distances \f$D_G\f$ and distances in the low-dimensional space \f$D_Y\f$. 
This metric requires knowledge of the geodesic distances, however. For
linear subspaces, the geodesic distances are given by the Euclidean
distances. Otherwise, a method for estimating the geodesic distances, 
such as the one provided in Isomap, must be utilized. 

To evaluate the distance residual using, for example, IsoMap:

\code
  int high_dim=20;
  int low_dim=3;
  
  ...
  
  redmet.find_t(input,output,low_dim,eigopts);
  
  // Dimensionalities to calculate the distance statistics for:
  int dimensionalities[3]={15, 10, 5}; 
  // Correlation coefficient between geodesic distance and low-euclidean
  double corrcoef[3];
  // Coefficient of determination (R^2) between geodesic distances and low
  double rsquared[3];
  // Wrappers
  YA_WRAP(int) wdim(dimensionalities,1,3);
  YA_WRAP(double) wcorr(corrcoef,1,3);
  YA_WRAP(double) wrsq(rsquared,1,3);
  // Calculate the statistics and store in vectors
  dist_residual(redmet.graph_dists(),output_matrix,dimensions,
                corrcoeff,rsquared);
                
  // -- or if landmarks are used:
  
  dist_residual(redmet.graph_dists(),output_matrix,dimensions,
                corrcoeff,rsquared,landmarks);  
\endcode

For PCA and MDS, the distance residual can be evaluated between the input
and output matrices used in the find calculation. For LLE and the autoencoder,
the distance residual is of little value.

As discussed earlier, a more general method that allows comparison between
different algorithms is the reconstruction error \ref ref "[Hinton, 2006]". To
calculate the reconstruction error using existing maps stored in redmet:

\code
  double err=redmet.reconstruct_error(input);
\endcode

Note that when using NRM for mapping (LLE, MDS, IsoMap), the reconstruction
error for the samples used to obtain the maps will always be 0.

The approaches listed above are often cited as methods for estimating
the intrinsic dimensionality of a manifold. However, they all
rely on dimensionality reduction methods that attempt an embedding
of sample data in a space with lower dimensionality. Therefore,
these approaches are really only suitable for estimating the smooth
\e embedding \e dimensionality. This subtlety is important because
the Whitney embedding theorem dictates that
a smooth embedding of a \f$d\f$-manifold may require as many as \f$2d+1\f$ 
dimensions. Knowledge of the smooth embedding dimensionality is desirable for
performing dimensionality reduction. For determining the 
\e intrinsic \e dimensionality, however, methods such as local-PCA might
be more accurate for manifolds with complex structure. This is 
because they do not rely on a single-coordinate embedding of the entire 
manifold.

Dr.L. provides a variant of local PCA, referred to as point PCA 
(see \ref ref "[Brown, 2009]") in order to estimate the intrinsic 
dimensionality. Taking the same approximations used in LLE and Isomap, 
we assume that a local region 
of a manifold given by a point and its nearest neighbors is approximately linear
(local PCA differs from point PCA in that generalized clustering techniques such 
as vector quantization are used to determine locality). This assumption allows for 
estimation of intrinsic dimensionality by assessing the error in fitting each set 
of points to a lower-dimensional hyperplane. PCA can be utilized to perform this 
task; for a \f$d\f$-dimensional manifold, the residual variance should be near zero
given an encoding with \f$d\f$ principal components. For example, in the case 
of a 2-dimensional manifold, the neighborhood of each point should reside on a 
2-dimensional plane and therefore the variance in the data should be explained 
entirely by the first 2 principal components.

For details on using point-PCA in Dr. L., see point_pca() for k-nearest
neighbors and point_pca_ep() for epsilon neighbors.

\n
\n
\section em Example Manifolds

A variety of sample manifolds (all embedded in 3 dimensions),
can be generated using Dr. L. as described
in the file ya_manifold_samples.h. For example, to generate the japanese
flag with 700 samples:

\code
  #include "dimred/ya_manifold_samples.h"

  int nS=700;
  double manifold[nS*3];
  int colors[nS];
  YA_WRAP(double) wmanifold(manifold,nS,3);
  
  // Set up a random number generator and seed it
  MathRandom<MathMersenneTwister> rng;
  unsigned long seed=123456789;
  rng.seed(seed);

  // Fill the matrix with the samples
  ya_manifold_japaneseflag(wmanifold,colors,nS,rng);
\endcode

\n
\n
\section advanced Advanced Use of Library

For examples on more advanced use of the library, see the cml_drl executable
supplied with Dr.L.

\n
\n
\section ref References

Brown, W.M., Martin, S., Pollock, S.N., Coutsias, E.A., Watson, J.-P. <B> Algorithmic
Dimensionality Reduction for Molecular Structure Analysis.</B> <I>Journal of Chemical
Physics</I>. 2009. 130: p. 044901.

Hinton, G.E., Salakhutdinov, R.R. <B>Reducing the Dimensionality of Data with
Neural Networks.</B> <I>Science</I>. 2006. 313: p.504-507.

Roweis, S.T., Saul, L.K. <B> Nonlinear Dimensionality Reduction by Locally Linear
Embedding.</B> <I>Science</I>. 2000. 290: p. 2323-2326.

Saul, L. Roweis, S. <B>Think Globally, Fit Locally: Unsupervised Learning of 
Low Dimensional Manifolds</B>. <I>Journal of Machine Learning Research</I>.
2003. 4: p.119-155.

Tenenbaum, J.B., de Silva, V., Langfor, J.C. <B>A Global Geometric Framework
for Nonlinear Dimensionality Reduction.</B> <I>Science</I>. 2000. 290: 
p. 2319-2323.

\n
\n
\n
\n

*/

#endif
