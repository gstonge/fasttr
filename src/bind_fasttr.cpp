#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <HistorySampler.hpp>

using namespace std;
using namespace fasttr;

namespace py = pybind11;


//template function to declare same class for different types of nodes
template<typename Node>
void declare_history_sampler(py::module &m, string typestr)
{
    string pyclass_name = typestr + string("HistorySampler");

    py::class_<HistorySampler<Node> >(m, pyclass_name.c_str())

        .def(py::init<const unordered_map<Node,unordered_set<Node>>&,
                const vector<long double>&, const vector<long double>&,
                unsigned int, long double, long double>(), R"pbdoc(
            Default constructor of the class.

            Args:
               adjacency_map: Adjacency map for the network. Must be a tree.
               kernel_vector: Vector of kernel value for attachement process.
               grad_kernel_vector: Vector of gradient kernel value according
                                   to some parameter.
               seed: Seed for the RNG.
               source_bias: Float for the exponent to bias toward more probable
                            source nodes. Default is 1, unbiased.
               sample_bias: Float for the exponent to bias toward more probable
                            nodes through the sampling. Default is 1, unbiased.

            )pbdoc", py::arg("adjacency_map"), py::arg("kernel_vector"),
                py::arg("grad_kernel_vector"), py::arg("seed") = 42,
                py::arg("source_bias") = 1., py::arg("sample_bias") = 1.)

        //accessors

        .def("get_adjacency", &HistorySampler<Node>::get_adjacency, R"pbdoc(
            Returns the adjacency map of the network.
            )pbdoc")

        .def("get_rooted_adjacency", &HistorySampler<Node>::get_rooted_adjacency,
                R"pbdoc(
            Returns the adjacency map of the network rooted at a source.
            )pbdoc")

        .def("get_probability", &HistorySampler<Node>::get_probability, R"pbdoc(
            Returns the map of probability for each node of being the source.
            )pbdoc")

        .def("get_histories", &HistorySampler<Node>::get_histories, R"pbdoc(
            Returns the vector of sampled histories.
            )pbdoc")

        .def("get_log_posterior", &HistorySampler<Node>::get_log_posterior,
                R"pbdoc(
            Returns a vector of log posterior for each sampled histories.
            )pbdoc")

        .def("get_grad_log_posterior",
                &HistorySampler<Node>::get_grad_log_posterior,
                R"pbdoc(
            Returns a vector of gradient on the log posterior, for each
            sampled histories, according to some parameter.
            )pbdoc")

        .def("get_ground_truth_log_posterior",
                &HistorySampler<Node>::get_ground_truth_log_posterior,
                R"pbdoc(
            Returns a long double for the log posterior of the ground truth.
            )pbdoc")

        .def("get_marginal_mean", &HistorySampler<Node>::get_marginal_mean,
                R"pbdoc(
            Returns a map of marginal mean arrival time for each node.
            )pbdoc")

        //mutators

        .def("root", &HistorySampler<Node>::root, R"pbdoc(
            Transform the network into a directed one rooted at source.

            Args:
               source: Node to root on.
            )pbdoc", py::arg("source"))

        .def("unroot", &HistorySampler<Node>::unroot, R"pbdoc(
            Revert the network into an undirected one.
            )pbdoc")

        .def("sample", &HistorySampler<Node>::sample, R"pbdoc(
            Sample uniformly histories.

            Args:
               nb_sample: Number of histories sampled.
            )pbdoc", py::arg("nb_sample"))

        .def("set_kernel_vector", &HistorySampler<Node>::set_kernel_vector,
                R"pbdoc(
            Change the kernel and grad kernel vectors for the attachement
            process.

            Args:
               kernel_vector: Vector for each degree representing the kernel.
               grad_kernel_vector: Vector for each degree representing the
                                   gradient of the kernel.
            )pbdoc", py::arg("kernel_vector"), py::arg("grad_kernel_vector"))

        .def("set_ground_truth", &HistorySampler<Node>::set_ground_truth,
                R"pbdoc(
            Set the ground truth associated to the network.

            Args:
               ground_truth: Vector of node label for the history.
            )pbdoc", py::arg("ground_truth"));
}

PYBIND11_MODULE(_fasttr, m)
{
    //functions

    //classes
    declare_history_sampler<std::string>(m, "String");
    declare_history_sampler<unsigned int>(m, "Int");
}
