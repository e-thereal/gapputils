/***************************************************************************
                                 listnode.h
                              W. Michael Brown
                             -------------------

    Node class with edges and weights in linked list

 __________________________________________________________________________

    Part of the Math Library
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

#ifndef LISTNODE_H
#define LISTNODE_H

#include <list>
#include <iostream>

//using namespace std;

YALANAMESPACE

/// Node object containing edges and weights to other nodes
/** - Insert/Delete in constant time
  * - Accessing an edge by id in O(n)
  * - \note Their is no check for duplicate edges when adding edges! **/
class ListNode {
 public:
  ListNode() {};
  ~ListNode() {};

  struct ListEdge;

  /// \var Pointer to an edge within a node
  typedef std::list<ListEdge>::iterator edge_pointer;

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
  std::list<ListEdge> edges;
};

struct ListNode::ListEdge {
  ya_sizet id;
  double weight;
};

// Add an edge
void ListNode::add_edge(const ya_sizet nodei, const double weight) {
  ListEdge e;
  e.id=nodei;
  e.weight=weight;
  edges.push_back(e);
}

// Return a pointer to first edge
ListNode::edge_pointer ListNode::begin() {
  return edges.begin();
}

// Return a pointer to end of edges
ListNode::edge_pointer ListNode::end() {
  return edges.end();
}

// Return the number of edges
ya_sizet ListNode::size() {
  return edges.size();
}

// Return the absolute node index of an edge
ya_sizet ListNode::node(edge_pointer ep) {
  return ep->id;
}

// Return the weight of an edge
double ListNode::weight(edge_pointer ep) {
  return ep->weight;
}

// Cut an edge based on the pointer
void ListNode::cut_edge(edge_pointer ep) {
  edges.erase(ep);
}
  
// Cut an edge based on the absolute id
void ListNode::cut_edge(const ya_sizet nodei) {
  for (edge_pointer ep=edges.begin(); ep!=edges.end(); ep++)
    if (ep->id==nodei) {
      edges.erase(ep);
      return;
    }
}


}
#endif

