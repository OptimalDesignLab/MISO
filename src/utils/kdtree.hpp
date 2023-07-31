#ifndef MISO_KDTREE
#define MISO_KDTREE

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "mfem.hpp"

namespace miso
{
/// Class for representing a point
/// \tparam coord_type - a numeric type
/// \tparam dim - number of spatial dim
template <typename coord_type, size_t dim>
class point
{
public:
   /// Copy constructor from mfem::Vector to point
   point(const mfem::Vector &x)
   {
      for (size_t i = 0; i < dim; ++i)
         coords_[i] = x(i);
   }

   /// Copy construction from point to point
   point(std::array<coord_type, dim> c) : coords_(c) { }

   /// Initializer list based constructor
   point(std::initializer_list<coord_type> list)
   {
      size_t n = std::min(dim, list.size());
      std::copy_n(list.begin(), n, coords_.begin());
   }

   /// Returns the coordinate in the given dimension.
   /// \param[in] index - dimension index (zero based)
   /// \returns coordinate in the given dimension
   coord_type get(size_t index) const { return coords_[index]; }

   /// Returns the distance squared from this point to another point
   /// \param[in] pt - another point
   /// \returns distance squared from this point to the other point
   double distance(const point &pt) const
   {
      double dist = 0;
      for (size_t i = 0; i < dim; ++i)
      {
         double d = get(i) - pt.get(i);
         dist += d * d;
      }
      return dist;
   }

private:
   std::array<coord_type, dim> coords_;
};

template <typename coord_type, size_t dim>
std::ostream &operator<<(std::ostream &out, const point<coord_type, dim> &pt)
{
   out << '(';
   for (size_t i = 0; i < dim; ++i)
   {
      if (i > 0) out << ", ";
      out << pt.get(i);
   }
   out << ')';
   return out;
}

/// k-d tree implementation, based on the version at rosettacode.org
/// \tparam coord_type - a numeric type
/// \tparam dim - number of spatial dim
template <typename coord_type, size_t dim>
class kdtree
{
public:
   typedef point<coord_type, dim> point_type;

private:
   /// Defines a node in the tree
   struct node
   {
      /// Construct a leaf in the tree
      /// \note this sets a default value of -1 for node_idx_
      node(const point_type &pt)
       : point_(pt), left_(nullptr), right_(nullptr), mfem_idx_(-1)
      { }
      /// Construct a leaf in a tree, including its index in the FE space.
      node(const mfem::Vector &x, int mfem_idx)
       : point_(x), left_(nullptr), right_(nullptr), mfem_idx_(mfem_idx)
      { }
      /// Get the `index` coordinate of the point
      coord_type get(size_t index) const { return point_.get(index); }
      /// Get the distance between the node and `pt`
      double distance(const point_type &pt) const
      {
         return point_.distance(pt);
      }
      point_type point_;
      node *left_;
      node *right_;
      int mfem_idx_;  /// maps back to node in mfem data structure
   };
   /// root node in the kd tree
   node *root_ = nullptr;
   /// current best node
   node *best_ = nullptr;
   /// current best distance
   double best_dist_ = 0;
   /// stored number nodes visited since last call of nearest
   size_t visited_ = 0;
   /// container for all nodes in the tree
   std::vector<node> nodes_;

   /// Defines an operator to compare two nodes based on coordinate `index_`
   struct node_cmp
   {
      node_cmp(size_t index) : index_(index) { }
      bool operator()(const node &n1, const node &n2) const
      {
         return n1.point_.get(index_) < n2.point_.get(index_);
      }
      size_t index_;
   };

   /// Recursive function to build a tree
   node *make_tree(size_t begin, size_t end, size_t index)
   {
      if (end <= begin) return nullptr;
      size_t n = begin + (end - begin) / 2;
      std::nth_element(
          &nodes_[begin], &nodes_[n], &nodes_[0] + end, node_cmp(index));
      index = (index + 1) % dim;
      nodes_[n].left_ = make_tree(begin, n, index);
      nodes_[n].right_ = make_tree(n + 1, end, index);
      return &nodes_[n];
   }

   /// Find the nearest node in tree to `point`
   void nearest(node *root, const point_type &point, size_t index)
   {
      if (root == nullptr) return;
      ++visited_;
      double d = root->distance(point);
      if (best_ == nullptr || d < best_dist_)
      {
         best_dist_ = d;
         best_ = root;
      }
      if (best_dist_ == 0) return;
      double dx = root->get(index) - point.get(index);
      index = (index + 1) % dim;
      nearest(dx > 0 ? root->left_ : root->right_, point, index);
      if (dx * dx >= best_dist_) return;
      nearest(dx > 0 ? root->right_ : root->left_, point, index);
   }

public:
   /// Remove the copy constructor and assignment constructors
   kdtree(const kdtree &) = delete;
   kdtree &operator=(const kdtree &) = delete;

   /// Construct an empty tree; use with set_size and add_node, and finalize
   kdtree() { }

   /// Constructor taking a pair of iterators. Adds each point in the range
   /// [begin, end) to the tree.
   /// \param[in] begin - start of range
   /// \param[in] end - end of range
   template <typename iterator>
   kdtree(iterator begin, iterator end) : nodes_(begin, end)
   {
      root_ = make_tree(0, nodes_.size(), 0);
   }

   /// Constructor taking a function object that generates points. The function
   /// /// object will be called n times to populate the tree. \param[in] f -
   /// function that returns a point \param[in] n - number of points to add
   template <typename func>
   kdtree(func &&f, size_t n)
   {
      nodes_.reserve(n);
      for (size_t i = 0; i < n; ++i)
         nodes_.emplace_back(f());
      root_ = make_tree(0, nodes_.size(), 0);
   }

   /// Allocate memory for the tree
   /// \param[in] n - number of nodes in the tree
   void set_size(size_t n) { nodes_.reserve(n); }

   /// Adds a new node to the tree, including its index
   /// \param[in] x - the coordinates of the node
   /// \param[in] mfem_idx - the index of the node in the finite-element space
   void add_node(const mfem::Vector &x, int mfem_idx)
   {
      nodes_.emplace_back(node(x, mfem_idx));
   }

   /// Call this after adding all the nodes to the tree
   void finalize() { root_ = make_tree(0, nodes_.size(), 0); }

   /// Returns true if the tree is empty, false otherwise.
   bool empty() const { return nodes_.empty(); }

   /// Returns the number of nodes visited by the last call to nearest()
   size_t visited() const { return visited_; }

   /// Returns the distance between the input point and return value from the
   /// last call to nearest()
   double distance() const { return std::sqrt(best_dist_); }

   /// Returns the index of the closest node.
   /// \warning based on the most recent call to nearest().
   // int get_node_index() const { return best_->node_idx_; }

   /// Returns the index of the element associated with the closest node.
   /// \warning based on the most recent call to nearest().
   // int get_elem_index() const { return best_->elem_idx_; }

   /// Finds the nearest point in the tree to the given point.
   /// \param[in] pt - a point whose distance to tree is sought
   /// \returns the nearest node (index) in the tree to the given point
   /// \note It is not valid to call this function if the tree is empty.
   int nearest(const point_type &pt)
   {
      if (root_ == nullptr) throw std::logic_error("tree is empty");
      best_ = nullptr;
      visited_ = 0;
      best_dist_ = 0;
      nearest(root_, pt, 0);
      return best_->mfem_idx_;
   }
};

#if 0
void test_wikipedia() {
    typedef point<int, 2> point2d;
    typedef kdtree<int, 2> tree2d;
 
    point2d points[] = { { 2, 3 }, { 5, 4 }, { 9, 6 }, { 4, 7 }, { 8, 1 }, { 7, 2 } };
 
    tree2d tree(std::begin(points), std::end(points));
    point2d n = tree.nearest({ 9, 2 });
 
    std::cout << "Wikipedia example data:\n";
    std::cout << "nearest point: " << n << '\n';
    std::cout << "distance: " << tree.distance() << '\n';
    std::cout << "nodes visited: " << tree.visited() << '\n';
}
 
typedef point<double, 3> point3d;
typedef kdtree<double, 3> tree3d;
 
struct random_point_generator {
    random_point_generator(double min, double max)
        : engine_(std::random_device()()), distribution_(min, max) {}
 
    point3d operator()() {
        double x = distribution_(engine_);
        double y = distribution_(engine_);
        double z = distribution_(engine_);
        return point3d({x, y, z});
    }
 
    std::mt19937 engine_;
    std::uniform_real_distribution<double> distribution_;
};
 
void test_random(size_t count) {
    random_point_generator rpg(0, 1);
    tree3d tree(rpg, count);
    point3d pt(rpg());
    point3d n = tree.nearest(pt);
 
    std::cout << "Random data (" << count << " points):\n";
    std::cout << "point: " << pt << '\n';
    std::cout << "nearest point: " << n << '\n';
    std::cout << "distance: " << tree.distance() << '\n';
    std::cout << "nodes visited: " << tree.visited() << '\n';
}
 
int main() {
    try {
        test_wikipedia();
        std::cout << '\n';
        test_random(1000);
        std::cout << '\n';
        test_random(1000000);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
#endif

}  // namespace miso

#endif