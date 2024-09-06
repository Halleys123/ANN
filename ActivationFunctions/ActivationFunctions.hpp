#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>

// here x means net_i

namespace ActivationFunctions {

    template <typename T>
    T u_sigmoid(T x, double lambda = 0.3) {
        return 1.0 / (1.0 + exp(-x * lambda));
    }

    template <typename T>
    T b_sigmoid(T x, double lambda = 0.3) {
        return (2.0 / (1.0 + exp(-x * lambda))) - 1.0;
    }
    
    template <typename T>
    T b_binary(T x) {
        return x > 0.0 ? 1.0 : (- 1.0);
    }
    
    template <typename T>
    T u_binary(T x) {
        return x > 0.0 ? 1.0 : (0.0);
    }  
    
    template <typename T>
    T swish(T x) {
        return x / (1 + exp(-x));
    }

    template <typename T>
    T relu(T x) {
        return x > 0 ? x : 0;
    }

    template <typename T>
    T tanh_func(T x) {
        return tanh(x);
    }

    template <typename T>
    T leaky_relu(T x, T alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }

    template <typename T>
    T linear(T x) {
        return x;
    }

}

#endif
