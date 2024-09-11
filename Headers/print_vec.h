#pragma once

#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vec) {
    os << "[\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << "  " << vec[i];
        if (i != vec.size() - 1)
            os << ",\n";
    }
    os << "\n]";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<std::vector<T>>>& vec) {
    os << "[\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << "Layer " << i + 1 << ":\n";
        os << vec[i];  // Use the 2D vector << operator
        if (i != vec.size() - 1)
            os << ",\n";
    }
    os << "\n]";
    return os;
}
