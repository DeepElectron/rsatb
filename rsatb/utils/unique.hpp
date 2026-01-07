#include <algorithm>
// using std::stable_sort, std::unique, std::unique_copy
#include <iterator>
// using std::back_inserter
#include <vector>
// using std::vector


/** Helper to implement argsort-like behavior */
template<class T>
struct SortPair
{
  T value;
  std::size_t index;

  SortPair(const T& value, std::size_t index)
    : value(value), index(index)
  {}
  SortPair(T&& value, std::size_t index)
    : value(std::move(value)), index(index)
  {}
  bool operator<(const SortPair& o) const
  { return value < o.value; }

  bool operator<(const T& o) const
  { return value < o; }

  friend bool operator<(const T& left, const SortPair& right)
  { return left < right.value; }

  bool operator==(const SortPair& o) const
  { return value == o.value; }

  friend void swap(SortPair& left, SortPair& right)
  {
      using std::swap;
      swap(left.value, right.value);
      swap(left.index, right.index);
  }
};
/**
 * Implements numpy.unique
 *
 * \tparam T scalar value type
 * \tparam Iterator input iterator for type T
 * \param first, last range of values
 * \param index if not null, returns the first indices of each unique value in
 *    the input range
 * \param inverse if not null, returns a mapping to reconstruct the input range
 *    from the output array. first[i] == returnvalue[inverse[i]]
 * \param count if not null, returns the number of times each value appears
 * \return sorted unique values from the input range
 */
template<class T, class Iterator>
std::vector<T> unique(Iterator first, Iterator last,
                      std::vector<std::size_t>* index=nullptr,
                      std::vector<std::size_t>* inverse=nullptr,
                      std::vector<std::size_t>* count=nullptr)
{
  std::vector<T> uvals;
  if(! (index || inverse || count)) { // simple case
    uvals.assign(first, last);
    using t_iter = typename std::vector<T>::iterator;
    const t_iter begin = uvals.begin(), end = uvals.end();
    std::sort(begin, end);
    uvals.erase(std::unique(begin, end), end);
    return uvals;
  }
  if(first == last) { // early out. Helps with inverse computation
    for(std::vector<std::size_t>* arg: {index, inverse, count})
      if(arg)
        arg->clear();
    return uvals;
  }
  using sort_pair = SortPair<T>;
  using sort_pair_iter = typename std::vector<sort_pair>::iterator;
  std::vector<sort_pair> sorted;
  for(std::size_t i = 0; first != last; ++i, ++first)
    sorted.emplace_back(*first, i);
  const sort_pair_iter sorted_begin = sorted.begin(), sorted_end = sorted.end();
  // stable_sort to keep first unique index
  std::stable_sort(sorted_begin, sorted_end);
  /*
   * Compute the unique values. If we still need the sorted original values,
   * this must be a copy. Otherwise we just reuse the sorted vector.
   * Note that the equality operator in SortPair only uses the value, not index
   */
  std::vector<sort_pair> upairs_copy;
  const std::vector<sort_pair>* upairs;
  if(inverse || count) {
    std::unique_copy(sorted_begin, sorted_end, std::back_inserter(upairs_copy));
    upairs = &upairs_copy;
  }
  else {
    sorted.erase(std::unique(sorted_begin, sorted_end), sorted_end);
    upairs = &sorted;
  }
  uvals.reserve(upairs->size());
  for(const sort_pair& i: *upairs)
    uvals.push_back(i.value);
  if(index) {
    index->clear();
    index->reserve(upairs->size());
    for(const sort_pair& i: *upairs)
      index->push_back(i.index);
  }
  if(count) {
    count->clear();
    count->reserve(uvals.size());
  }
  if(inverse) {
    inverse->assign(sorted.size(), 0);
    // Since inverse->resize assigns index 0, we can skip the 0-th outer loop
    sort_pair_iter sorted_i =
      std::upper_bound(sorted_begin, sorted_end, uvals.front());
    if(count)
      count->push_back(std::distance(sorted_begin, sorted_i));
    for(std::size_t i = 1; i < uvals.size(); ++i) {
      const T& uval = uvals[i];
      const sort_pair_iter range_start = sorted_i;
      // we know there is at least one value
      do
        (*inverse)[sorted_i->index] = i;
      while(++sorted_i != sorted_end && sorted_i->value == uval);
      if(count)
        count->push_back(std::distance(range_start, sorted_i));
    }
  }
  if(count && ! inverse) {
    sort_pair_iter range_start = sorted_begin;
    for(const T& uval: uvals) {
      sort_pair_iter range_end =
        std::find_if(std::next(range_start), sorted_end,
                     [&uval](const sort_pair& i) -> bool {
                       return i.value != uval;
                     });
      count->push_back(std::distance(range_start, range_end));
      range_start = range_end;
    }
    /*
     * We could use equal_range or a custom version based on an
     * exponential search to reduce the number of comparisons.
     * The reason we don't use equal_range is because it has worse
     * performance in the worst case (uvals.size() == sorted.size()).
     * We could make a heuristic and dispatch between both implementations
     */
  }
  return uvals;
}