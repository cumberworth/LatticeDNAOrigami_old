// utility.h

#ifndef UTILITY_H
#define UTILITY_H

namespace Utility {
    template<typename Container_T, typename Element_T>
    int index(Container_T container, Element_T element);

    struct NoElement {};
}

/* Copied from a stack exchange question (which is copied from the BOOST
   library) for allowing pairs to be hashed.
*/
template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {

    template<typename S, typename T> struct hash<pair<S, T>> {
        inline size_t operator()(const pair<S, T>& v) const {
            size_t seed = 0;
            ::hash_combine(seed, v.first);
            ::hash_combine(seed, v.second);
            return seed;
        }
    };

    template<typename T> struct hash<vector<T>> {
        inline size_t operator()(const vector<T>& v) const {
            size_t seed = 0;
            for (auto i: v) {
                ::hash_combine(seed, i);
            }
            return seed;
        }
    };
}

#endif // UTILITY_H
