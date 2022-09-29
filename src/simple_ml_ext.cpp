#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    float logit[k];
    float grad_theta[n * k];
    // std::cout << "m=" << m << "\tn=" << n << "\tk=" << k << std::endl;
    for(size_t index=0; index <= m / batch; index++) {
        size_t row_s = index * batch;
        size_t row_e = row_s + batch > m ? m : row_s + batch;
        size_t rown = row_e - row_s;
        if (rown <= 0) break;
        // std::cout << "index:\t" << index << std::endl;
        // forward_pass
        for (size_t ix = 0; ix < n * k; ix++) {
            grad_theta[ix] = 0.0;
        }
        for (size_t row_i = row_s;row_i < row_e; row_i ++) {
            float cum_sum = 0.0;
            for (size_t k_i = 0; k_i < k; k_i++) {
                logit[k_i] = 0.0;
                for (size_t j=0;j<n;j++) {
                    logit[k_i] += X[row_i * n + j] * theta[j * k + k_i];
                    // std::cout << X[row_i * n + j] << "----" << theta[j * n + k] << logit[k_i] << std::endl;
                }
                logit[k_i] = std::exp(logit[k_i]);
                cum_sum += logit[k_i];
                // std::cout << logit[k_i] << "\t++++\t cum_sum=" << cum_sum << std::endl;
            }
            // std::cout << "label: " << uint(y[row_i]) << std::endl;
            for (size_t k_i = 0; k_i < k; k_i++) {
                logit[k_i] = logit[k_i] / cum_sum;
                // std::cout << "row_i: \t" << row_i << "\tlogit :\t" << k_i << "=" << logit[k_i] << std::endl;
            }
            for (size_t k_i = 0; k_i < k; k_i++) {
                for (size_t j = 0; j < n; j++) {
                    grad_theta[j * k + k_i] +=  X[row_i * n + j] * (logit[k_i] - (uint(y[row_i]) == k_i ? 1.0 : 0));
                    // std::cout << "k_i:\t" << k_i << "\tj:\t" << j << "\t" << X[row_i * n + j] * (logit[k_i] - (uint(y[row_i]) == k_i ? 1.0 : 0)) << "~~~~" << grad_theta[j * k + k_i] << std::endl;
                }
            }
            // if (row_i >= 4) break;
        }
        for (size_t ix = 0; ix < n * k; ix++) {
            theta[ix] -= grad_theta[ix] * lr / rown;
        }
    }

}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
