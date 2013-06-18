/***************************************************************************
                                 namegraph.h
                             -------------------
                               W. Michael Brown

  Base Graph Container

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Oct 24 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef NAMEGRAPH_H
#define NAMEGRAPH_H
 
#include <stdlib.h> 
#include <vector>
#include <limits>
#include "listnode.h"
#include "mapnode.h"
//using namespace std;

YALANAMESPACE

/// Base Graph Container
template<class ntype>
class NameGraph {
 public:
  /// Empty graph
  NameGraph();
  /// Graph with a specified number of nodes
  NameGraph(const ya_sizet numnodes);
  /// Graph with a specified number of nodes, each with a reserve for num_edges
  NameGraph(const ya_sizet numnodes, const ya_sizet numedges);
  ~NameGraph();

  /// Add a node
  void addnode();
  /// Add an edge
  void add_edge(const ya_sizet node1, const ya_sizet node2, 
                const double weight);
  /// Add a k-nearest neighbors matrix of edges and weights equal to distances
  /** The matrices have nodes for rows and neighbors for columns **/
  void addneighborhood(const YA_MatI &neighbors, const YA_MatD &distances);  
  /// Add a epsilon-nearest neighbors matrix of edges and weights (distances)
  /** The matrices have nodes for rows and neighbors for columns **/
  void addneighborhood(const std::vector<YA_DynI> &neighbors,
                       const std::vector<YA_DynD> &distances);

  /// Access a node
  ntype & operator[](const ya_sizet nodenum);
  
  /// Return the size
  inline ya_sizet size() const { return graph.size(); }

  /// Calculate shortest path from start to all other nodes
  void dijkstra(const ya_sizet start, std::vector<double> &dist);
  /// Calculate a pairwise distance matrix based on shortest paths
  /** std::numeric_limits<ntype>::max() is assigned for unconnected nodes 
    * \param verbose if non-zero, a progress meter and time info to console **/
  void all_shortest_paths(VM_SymMatD &pair_dists, const int verbose);
  /// Calculate a distance matrix from source_nodes to all nodes in graph
  /** \param source_nodes list of source indices in graph (each < graph.size())
    * \param pair_dists returns a source_nodes.numel() x graph.size() matrix
    *        of distances 
    * \param verbose if non-zero, a progress meter and time info to console **/
  template<class vm_type, class ya_type2>
  void shortest_paths(const vm_type &source_nodes, ya_type2 &pair_dists,
                      const int verbose);

  /// Write out the edges to a file
  /** The format is:
    * \verbatim node1 node2 weight \endverbatim **/
  void writeedges(const std::string &filename);
 
 protected:
   std::vector<ntype> graph;
};

///\var typedef NameGraph<ListNode> ListGraph
typedef NameGraph<ListNode> ListGraph;
///\var typedef NameGraph<MapNode> MapGraph
typedef NameGraph<MapNode> MapGraph;

}
#endif
