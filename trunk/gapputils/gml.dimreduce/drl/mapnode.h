/***************************************************************************
                                  mapnode.h
                              W. Michael Brown
                             -------------------

    Node class with edges and weights in map

 __________________________________________________________________________

    Part of the Math Library
 __________________________________________________________________________

    begin      : Tue Nov 14 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef MAPNODE_H
#define MAPNODE_H

#include <map>
#include <iostream>

//using namespace std;

YALANAMESPACE

/// Node object containing edges and weights to other nodes
/** - Insert/Delete in log(n) time
  * - Accessing an edge by id in log(n)
  * - Multiple edges to the same node will never exist **/
class MapNode {
 public:
  MapNode() {}
  ~MapNode() {}

  struct MapEdge;

  /// \var Pointer to an edge within a node
  typedef std::map<ya_sizet,MapEdge>::iterator edge_pointer;

  /// Add an edge
  inline void add_edge(const ya_sizet nodei, const double weight);
  /// Return a pointer to first edge
  inline edge_pointer begin();
  /// Return a pointer to end of edges
  inline edge_pointer end();
  /// Return the absolute node index of an edge
  inline ya_sizet node(edge_pointer ep);
  /// Return the weight of an edge
  inline double weight(edge_pointer ep);
  /// Return the number of edges
  inline ya_sizet size();
  /// Cut an edge based on the pointer
  inline void cut_edge(edge_pointer ep);
  /// Cut an edge based on the absolute id
  inline void cut_edge(const ya_sizet nodei);

 private:
  std::map<ya_sizet,MapEdge> edges;
};

struct MapNode::MapEdge {
  MapEdge(void){weight=0.0;}
  double weight;
};

// Add an edge
void MapNode::add_edge(const ya_sizet nodei, const double weight) {
  MapEdge e;
  edges[nodei]=e;
  edges[nodei].weight=weight;
}

// Return a pointer to first edge
MapNode::edge_pointer MapNode::begin() {
  return edges.begin();
}

// Return a pointer to end of edges
MapNode::edge_pointer MapNode::end() {
  return edges.end();
}

// Return the number of edges
ya_sizet MapNode::size() {
  return edges.size();
}

// Return the absolute node index of an edge
ya_sizet MapNode::node(edge_pointer ep) {
  return ep->first;
}

// Return the weight of an edge
double MapNode::weight(edge_pointer ep) {
  return ep->second.weight;
}

// Cut an edge based on the pointer
void MapNode::cut_edge(edge_pointer ep) {
  edges.erase(ep);
}

// Cut an edge based on the absolute id
void MapNode::cut_edge(const ya_sizet nodei) {
  edge_pointer ep=edges.find(nodei);
  edges.erase(ep);
}

}
#endif

