//===- DiscretePDF.inc - --*- C++ -*---------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>
namespace klee {

template <class T>
class DiscretePDF<T>::Node
{
private:
  bool m_mark;

public:
  Node *parent, *left, *right;
  T key;
  weight_type weight, sumWeights;

public:
  Node(T key_, weight_type weight_, Node *parent_);
  ~Node();

  Node *sibling() { return this==parent->left?parent->right:parent->left; }

  void markRed() { m_mark = true; }
  void markBlack() { m_mark = false; }
  bool isBlack() { return !m_mark; }
  bool leftIsBlack() { return !left || left->isBlack(); }
  bool rightIsBlack() { return !right || right->isBlack(); }
  void setSum() { 
    sumWeights = weight;
    if (left) sumWeights += left->sumWeights;
    if (right) sumWeights += right->sumWeights;
  }
};

  ///

template <class T>
DiscretePDF<T>::Node::Node(T key_, weight_type weight_, Node *parent_) {
  m_mark = false;

  key = key_;
  weight = weight_;
  sumWeights = 0;
  left = right = 0;
  parent = parent_;
}

template <class T>
DiscretePDF<T>::Node::~Node() {
  delete left;
  delete right;
}

//

template <class T>
DiscretePDF<T>::DiscretePDF() {
  m_root = 0;
}

template <class T>
DiscretePDF<T>::~DiscretePDF() {
  delete m_root;
}

template <class T>
bool DiscretePDF<T>::empty() const {
  return m_root == 0;
}

template <class T>
void DiscretePDF<T>::insert(T item, weight_type weight) {
  Node *p=0, *n=m_root;

  while (n) {
    if (!n->leftIsBlack() && !n->rightIsBlack())
      split(n);

    p = n;
    if (n->key==item) {
      assert(0 && "insert: argument(item) already in tree");
    } else {
      n = (item<n->key)?n->left:n->right;
    }
  }

  n = new Node(item, weight, p);

  if (!p) {
    m_root = n;
  } else {
    if (item<p->key) {
      p->left = n;
    } else {
      p->right = n;
    }

    split(n);
  }

  propogateSumsUp(n);
}

template <class T>
void DiscretePDF<T>::remove(T item) {
  Node **np = lookup(item, 0);
  Node *child, *n = *np;

  if (!n) {
    assert(0 && "remove: argument(item) not in tree");
  } else {
    if (n->left) {
      Node **leftMaxp = &n->left;

      while ((*leftMaxp)->right)
        leftMaxp = &(*leftMaxp)->right;

      n->key = (*leftMaxp)->key;
      n->weight = (*leftMaxp)->weight;

      np = leftMaxp;
      n = *np;
    }

    // node now has at most one child

    child = n->left?n->left:n->right;
    *np = child;

    if (child) {
      child->parent = n->parent;

      if (n->isBlack()) {
        lengthen(child);
      }
    }

    propogateSumsUp(n->parent);

    n->left = n->right = 0;
    delete n;
  }
}

template <class T>
void DiscretePDF<T>::update(T item, weight_type weight) {
  Node *n = *lookup(item, 0);

  if (!n) {
    assert(0 && "update: argument(item) not in tree");
  } else {
    n->weight = weight;
    propogateSumsUp(n);
  }
}

template <class T>
T DiscretePDF<T>::choose(double p) {
  assert (!((p < 0.0) || (p >= 1.0)) && "choose: argument(p) outside valid range");

  if (!m_root)
    assert(0 && "choose: choose() called on empty tree");

  weight_type w = (weight_type) (m_root->sumWeights * p);
  Node *n = m_root;

  while (1) {
    if (n->left) {
      if (w<n->left->sumWeights) {
	n = n->left;
	continue;
      } else {
	w -= n->left->sumWeights;
      }
    }
    if (w<n->weight || !n->right) {
      break; // !n->right condition shouldn't be necessary, just sanity check
    }
    w -= n->weight;
    n = n->right;
  }

  return n->key;
}



// template <class T>
// T DiscretePDF<T>::choose(double p) {
//   if (!m_root)
//     assert(0 && "choose: choose() called on empty tree");

//   Node *n = m_root;
//   Node *maxNode = n;  // Start with the root as the maximum node
//   weight_type maxWeight = n->weight;  // Initialize maxWeight with the root node's weight

//   // Use a stack to implement non-recursive in-order traversal
//   std::stack<Node*> stack;
//   while (!stack.empty() || n != nullptr) {
//     // Reach the left most Node of the current Node
//     while (n != nullptr) {
//       stack.push(n);
//       n = n->left;
//     }

//     // Current must be NULL at this point
//     n = stack.top();
//     stack.pop();

//     // Check if the current node has a greater weight
//     if (n->weight > maxWeight) {
//       maxWeight = n->weight;
//       maxNode = n;
//     }

//     // Move to the right node
//     n = n->right;
//   }

//   return maxNode->key;  // Return the key of the node with the highest weight
// }

template <class T>
bool DiscretePDF<T>::inTree(T item) {
  Node *n = *lookup(item, 0);

  return !!n;
}

template <class T>
typename DiscretePDF<T>::weight_type DiscretePDF<T>::getWeight(T item) {
  Node *n = *lookup(item, 0);
  assert(n);
  return n->weight;
}

//

template <class T>
typename DiscretePDF<T>::Node **
DiscretePDF<T>::lookup(T item, Node **parent_out) {
  Node *n, *p=0, **np=&m_root;

  while ((n = *np)) {
    if (n->key==item) {
      break;
    } else {
      p = n;
      if (item<n->key) {
        np = &n->left;
      } else {
        np = &n->right;
      }
    }
  }

  if (parent_out)
    *parent_out = p;
  return np;
}

template <class T>
void DiscretePDF<T>::split(Node *n) {
  if (n->left) n->left->markBlack();
  if (n->right) n->right->markBlack();

  if (n->parent) {
    Node *p = n->parent;

    n->markRed();

    if (!p->isBlack()) {
      p->parent->markRed();

      // not same direction
      if (!((n==p->left && p==p->parent->left) || 
            (n==p->right && p==p->parent->right))) {
        rotate(n);
        p = n;
      }

      rotate(p);
      p->markBlack();
    }
  }
}

template <class T>
void DiscretePDF<T>::rotate(Node *n) {
  Node *p=n->parent, *pp=p->parent;

  n->parent = pp;
  p->parent = n;

  if (n==p->left) {
    p->left = n->right;
    n->right = p;
    if (p->left) p->left->parent = p;
  } else {
    p->right = n->left;
    n->left = p;
    if (p->right) p->right->parent = p;
  }

  n->setSum();
  p->setSum();
	
  if (!pp) {
    m_root = n;
  } else {
    if (p==pp->left) {
      pp->left = n;
    } else {
      pp->right = n;
    }
  }
}

template <class T>
void DiscretePDF<T>::lengthen(Node *n) {
  if (!n->isBlack()) {
    n->markBlack();
  } else if (n->parent) {
    Node *sibling = n->sibling();

    if (sibling && !sibling->isBlack()) {
      n->parent->markRed();
      sibling->markBlack();

      rotate(sibling); // node sibling is now old sibling child, must be black
      sibling = n->sibling();
    }

    // sibling is black

    if (!sibling) {
      lengthen(n->parent);
    } else if (sibling->leftIsBlack() && sibling->rightIsBlack()) {
      if (n->parent->isBlack()) {
        sibling->markRed();
        lengthen(n->parent);
      } else {
        sibling->markRed();
        n->parent->markBlack();
      }
    } else {
      if (n==n->parent->left && sibling->rightIsBlack()) {
        rotate(sibling->left); // sibling->left must be red
        sibling->markRed();
        sibling->parent->markBlack();
        sibling = sibling->parent;
      } else if (n==n->parent->right && sibling->leftIsBlack()) {
        rotate(sibling->right); // sibling->right must be red
        sibling->markRed();
        sibling->parent->markBlack();
        sibling = sibling->parent;
      }

      // sibling is black, and sibling's far child is red

      rotate(sibling);
      if (!n->parent->isBlack()) 
        sibling->markRed();
      sibling->left->markBlack();
      sibling->right->markBlack();
    }
  }
}

template <class T>
void DiscretePDF<T>::propogateSumsUp(Node *n) {
  for (; n; n=n->parent)
    n->setSum();
}

}

