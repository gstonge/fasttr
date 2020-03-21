#ifndef HISTORYSAMPLER_HPP_
#define HISTORYSAMPLER_HPP_

#include <iostream>
#include "SamplableSet/SamplableSetCR.hpp"
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <deque>

namespace fasttr
{//start of namespace fasttr

const long double INF = std::numeric_limits<long double>::infinity();

/*
 * Class to perform fast sampling of history of growing trees
 */
template <typename Node>
class HistorySampler
{
public:
    //typedef std::pair<Node,Node> Edge;
    typedef std::unordered_map<Node,std::unordered_set<Node>> AdjacencyMap;
    typedef std::vector<Node> History;

    //Default constructor
    HistorySampler(
            const std::unordered_map<Node,std::unordered_set<Node>>& adjacency_map,
            const std::vector<long double>& kernel_vector,
            const std::vector<long double>& grad_kernel_vector,
            unsigned int seed = 42, long double source_bias = 1, long double sample_bias = 1);

    //Accessors
    const AdjacencyMap& get_adjacency() const
        {return adjacency_map_;}
    const AdjacencyMap& get_rooted_adjacency() const
        {return rooted_adjacency_map_;}
    const std::unordered_map<Node,long double>& get_probability() const
        {return probability_map_;}
    const std::vector<History>& get_histories() const
        {return history_vector_;}
    const std::vector<long double>& get_log_posterior() const
        {return log_posterior_vector_;}
    const std::vector<long double>& get_grad_log_posterior() const
        {return grad_log_posterior_vector_;}
    long double get_ground_truth_log_posterior();
    std::unordered_map<Node,long double> get_marginal_mean();

    //Mutators
    void root(const Node& source);
    void unroot();

    void sample(std::size_t nb_sample);
    void set_kernel_vector(const std::vector<long double>& kernel_vector,
            const std::vector<long double>& grad_kernel_vector)
        {kernel_vector_ = kernel_vector;
         grad_kernel_vector_ = grad_kernel_vector;
         update_log_posterior();}
    void set_ground_truth(const History& ground_truth)
        {ground_truth_ = ground_truth;}


private:
    AdjacencyMap adjacency_map_;
    AdjacencyMap rooted_adjacency_map_;
    std::vector<long double> kernel_vector_;
    std::vector<long double> grad_kernel_vector_;
    std::size_t n_;
    mutable sset::SamplableSetCR<Node> source_;
    mutable sset::SamplableSetCR<Node> boundary_;
    Node root_;
    bool rooted_;
    std::unordered_map<Node,
        std::unordered_map<Node,std::size_t>> descendant_map_;
    std::unordered_map<Node,long double> probability_map_; //for source
    std::vector<History> history_vector_;
    std::vector<long double> log_posterior_vector_; //for history
    std::vector<long double> grad_log_posterior_vector_; //for history
    std::vector<long double> log_posterior_bias_vector_; //for history
    History ground_truth_;
    long double source_bias_;
    long double sample_bias_;
    long double log_number_of_histories_;

    //methods
    std::size_t compute_descendant(const Node& node);
    void compute_source_probability(const Node& node);
    void compute_number_of_histories();

    std::pair<long double,long double> compute_log_posterior(const History& history);
    void update_log_posterior();
};

//Default constructor
template <typename Node>
HistorySampler<Node>::HistorySampler(
        const std::unordered_map<Node,std::unordered_set<Node>>& adjacency_map,
        const std::vector<long double>& kernel_vector,
        const std::vector<long double>& grad_kernel_vector,
        unsigned int seed, long double source_bias, long double sample_bias) :
    adjacency_map_(adjacency_map),
    rooted_adjacency_map_(),
    kernel_vector_(kernel_vector),
    grad_kernel_vector_(grad_kernel_vector),
    n_(adjacency_map_.size()),
    boundary_(std::min(static_cast<long double>(1.),pow(static_cast<long double>(n_),sample_bias)),
            std::max(static_cast<long double>(1.),pow(static_cast<long double>(n_),sample_bias)),seed),
    source_(1,1,seed+1),
    root_(),
    rooted_(false),
    descendant_map_(),
    probability_map_(),
    history_vector_(),
    log_posterior_vector_(),
    grad_log_posterior_vector_(),
    log_posterior_bias_vector_(),
    ground_truth_(),
    source_bias_(source_bias),
    sample_bias_(sample_bias),
    log_number_of_histories_(0.)
{
    //first node is chosen as the source to determine the probability
    //of each node being the source
    auto it = adjacency_map_.begin();
    const Node& source = (*it).first;

    //compute descendant
    root(source);
    compute_descendant(source);
    probability_map_[source] = 1.;
    compute_source_probability(source);
    unroot();

    //compute log number of histories
    compute_number_of_histories();

    //normalize and determine min max biased probabilities
    long double weight_sum = 0.;
    for (auto& element : probability_map_)
    {
        element.second = pow(element.second,source_bias_);//apply bias
        weight_sum += element.second;
    }
    long double min_probability = 1.;
    long double max_probability = 0.;
    long double p;
    for (auto& element : probability_map_)
    {
        element.second /= weight_sum;
        p = element.second;
        if (p < min_probability)
        {
            min_probability = p;
        }
        if (p > max_probability)
        {
            max_probability = p;
        }
    }

    //create samplable set for source
    source_ = sset::SamplableSetCR<Node>(
            min_probability,max_probability,seed+1);
    for (const auto& element : probability_map_)
    {
        source_.insert(element.first,element.second);
    }
}

//root the network at source using bfs
template <typename Node>
void HistorySampler<Node>::root(const Node& source)
{
    root_ = source;
    std::unordered_set<Node> reached;
    reached.insert(source);
    rooted_adjacency_map_[source] = std::unordered_set<Node>();
    std::deque<std::vector<Node>> queue(1);
    for (const auto& neighbor : adjacency_map_[source])
    {
        rooted_adjacency_map_[source].insert(neighbor);
        queue.back().push_back(neighbor);
        reached.insert(neighbor);
    }
    while (queue.size() > 0)
    {
        if (queue.front().size() > 0)
        {
            queue.emplace_back();
            for (const auto& node : queue.front())
            {
                for (const auto& neighbor : adjacency_map_[node])
                {
                    if (reached.count(neighbor) == 0)
                    {
                        reached.insert(neighbor);
                        rooted_adjacency_map_[node].insert(neighbor);
                        queue.back().push_back(neighbor);
                    }
                }
            }
        }
        queue.pop_front();
    }
    rooted_ = true;
}

//unroot the current source
template <typename Node>
void HistorySampler<Node>::unroot()
{
    rooted_adjacency_map_.clear();
    root_ = Node(); //default
    rooted_ = false;
}

//recursive function to calculate the number of descendant of that node
template <typename Node>
std::size_t HistorySampler<Node>::compute_descendant(const Node& node)
{
    if (rooted_)
    {
        descendant_map_[root_][node] = 1;
        for (const auto& neighbor: rooted_adjacency_map_[node])
        {
            descendant_map_[root_][node] += compute_descendant(neighbor);
        }
    }
    else
    {
        throw std::invalid_argument("Must be rooted");
    }
    return descendant_map_[root_].at(node);
}

//recursive function to calculate the probability of each node
//to be the source
template <typename Node>
void HistorySampler<Node>::compute_source_probability(const Node& node)
{
    if (rooted_)
    {
        for (const auto& neighbor : rooted_adjacency_map_[node])
        {
            probability_map_[neighbor] = (
                    probability_map_[node]*descendant_map_[root_][neighbor]/
                    (n_ - descendant_map_[root_][neighbor]));
            compute_source_probability(neighbor);
        }
    }
    else
    {
        throw std::invalid_argument("Must be rooted");
    }
}

//calculate the log number of possible histories
//simply by tracking a random history probability
template <typename Node>
void HistorySampler<Node>::compute_number_of_histories()
{
    log_number_of_histories_ = 0.;
    auto it = adjacency_map_.begin();
    const Node& source = (*it).first;
    std::vector<Node> boundary;

    long double weight_sum = 0.;
    for (auto& element : probability_map_)
    {
        weight_sum += element.second;
    }
    log_number_of_histories_ += log(weight_sum/probability_map_[source]);
    root(source);
    if (descendant_map_.count(source) == 0)
    {
        compute_descendant(source);
    }
    std::unordered_map<Node,
        std::size_t>& descendant_ = descendant_map_.at(source);
    //add neighbors of root to boundary
    weight_sum = 0.;
    for (const auto& neighbor : rooted_adjacency_map_[source])
    {
        boundary.push_back(neighbor);
        weight_sum += descendant_[neighbor];
    }
    while (boundary.size() > 0)
    {
        Node node = boundary.back();
        //compute bias relative to uniform
        log_number_of_histories_ += log(weight_sum/descendant_[node]);
        weight_sum -= descendant_[node];
        boundary.pop_back();
        //add neighbors of node to boundary
        for (const auto& neighbor : rooted_adjacency_map_[node])
        {
            boundary.push_back(neighbor);
            weight_sum += descendant_[neighbor];
        }
    }
    unroot();
}

//sample a certain number of histories
template <typename Node>
void HistorySampler<Node>::sample(std::size_t nb_sample)
{
    if (nb_sample <= 0)
    {
        throw std::runtime_error("The sample size must be superior to 0");
    }
    long double min_weight = std::min(static_cast<long double>(1.),pow(static_cast<long double>(n_),sample_bias_));
    long double max_weight = std::max(static_cast<long double>(1.),pow(static_cast<long double>(n_),sample_bias_));
    if (max_weight/min_weight > 1e16)
    {
        throw std::runtime_error("min and max weigth span too many scales (>1e16)");
    }

    history_vector_.clear();
    log_posterior_bias_vector_.clear();

    long double log_posterior_bias; //log posterior prob for the history due to bias
    for (std::size_t i = 0; i < nb_sample; i++)
    {
        log_posterior_bias = log_number_of_histories_;
        history_vector_.push_back(History());
        History& history = history_vector_[i];
        std::pair<Node,long double> node_prob = source_.sample();
        Node source = node_prob.first;
        //compute bias relative to uniform
        log_posterior_bias += log(node_prob.second/source_.total_weight());
        history.push_back(source);
        root(source);
        if (descendant_map_.count(source) == 0)
        {
            compute_descendant(source);
        }
        std::unordered_map<Node,
            std::size_t>& descendant_ = descendant_map_.at(source);
        //add neighbors of root to boundary
        for (const auto& neighbor : rooted_adjacency_map_[source])
        {
            boundary_.insert(neighbor, pow(static_cast<long double>(descendant_[neighbor]),sample_bias_));
        }
        while (boundary_.size() > 0)
        {
            node_prob = boundary_.sample();
            Node node = node_prob.first;
            //compute bias relative to uniform
            log_posterior_bias += log(node_prob.second/boundary_.total_weight());
            boundary_.erase(node);
            history.push_back(node);
            //add neighbors of node to boundary
            for (const auto& neighbor : rooted_adjacency_map_[node])
            {
                boundary_.insert(neighbor, pow(static_cast<long double>(descendant_[neighbor]),sample_bias_));
            }
        }
        boundary_.clear();
        unroot();
        log_posterior_bias_vector_.push_back(log_posterior_bias);
    }
    //get the log probability associated with each history
    update_log_posterior();
}

//get log probability of an history, and the gradient of that according to
//some parameter
template <typename Node>
std::pair<long double,long double> HistorySampler<Node>::compute_log_posterior(
        const History& history)
{
    std::unordered_map<Node,std::size_t> degree_map;
    degree_map[history[0]] = 1;
    degree_map[history[1]] = 1;
    long double Z = 2*kernel_vector_[1];
    long double dZ = 2*grad_kernel_vector_[1];
    long double log_prob = 0.;
    long double grad_log_prob = 0.;
    for (std::size_t t = 2 ; t < history.size(); t++)
    {
        const Node& node = history[t];
        degree_map[node] = 0;
        for (const auto& neighbor : adjacency_map_[node])
        {
            if (degree_map.count(neighbor) > 0)
            {
                log_prob += log(kernel_vector_[degree_map[neighbor]]);
                log_prob -= log(Z);
                grad_log_prob += (grad_kernel_vector_[degree_map[neighbor]]
                        /kernel_vector_[degree_map[neighbor]]);
                grad_log_prob -= dZ/Z;
            }
        }
        for (const auto& neighbor : adjacency_map_[node])
        {
            if (degree_map.count(neighbor) > 0)
            {
                Z -= kernel_vector_[degree_map[neighbor]];
                dZ -= grad_kernel_vector_[degree_map[neighbor]];
                degree_map[neighbor] += 1;
                degree_map[node] += 1;
                Z += kernel_vector_[degree_map[neighbor]];
                dZ += grad_kernel_vector_[degree_map[neighbor]];
            }
        }
        Z += kernel_vector_[degree_map[node]];
        dZ += grad_kernel_vector_[degree_map[node]];
    }
    return std::make_pair(log_prob,grad_log_prob);
}


template <typename Node>
void HistorySampler<Node>::update_log_posterior()
{
    if (history_vector_.size() <= 0)
    {
        throw std::runtime_error("The sample size must be superior to 0");
    }

    log_posterior_vector_.clear();
    log_posterior_vector_.reserve(history_vector_.size());
    grad_log_posterior_vector_.clear();
    grad_log_posterior_vector_.reserve(history_vector_.size());

    //get log probability
    for (int i = 0; i < history_vector_.size(); i++)
    {
        std::pair<long double,long double> grad_log_prob_pair = compute_log_posterior(
                history_vector_[i]);
        //we account the for the bias sampling in the posterior
        log_posterior_vector_.push_back(
                grad_log_prob_pair.first - log_posterior_bias_vector_[i]);
        grad_log_posterior_vector_.push_back(grad_log_prob_pair.second);
    }
}


//return a map of the marginal mean arrival time for each node
template <typename Node>
std::unordered_map<Node,long double> HistorySampler<Node>::get_marginal_mean()
{
    std::unordered_map<Node,long double> marginal_mean;
    long double max_log_prob = *std::max_element(log_posterior_vector_.begin(),
            log_posterior_vector_.end());
    std::vector<long double> probability_vector(log_posterior_vector_);
    //get weight
    long double total_prob = 0;
    for (auto& prob : probability_vector)
    {
        prob = exp(prob - max_log_prob);
        total_prob += prob;
    }
    //normalize
    for (auto& prob : probability_vector)
    {
        prob /= total_prob;
    }
    //calculate mean marginal for each node
    for (std::size_t i = 0; i < history_vector_.size(); i++)
    {
        History& history = history_vector_[i];
        for (std::size_t t = 0; t < history.size(); t++)
        {
            Node& node = history[t];
            if (marginal_mean.count(node) == 0)
            {
                marginal_mean[node] = t*probability_vector[i];
            }
            else
            {
                marginal_mean[node] += t*probability_vector[i];
            }
        }
    }
    return marginal_mean;
}

template <typename Node>
long double HistorySampler<Node>::get_ground_truth_log_posterior()
{
    if (ground_truth_.size() > 0)
    {
        return compute_log_posterior(ground_truth_).first;
    }
    else
    {
        throw std::runtime_error("No ground_truth set");
    }
}

}//end of namespace fasttr






#endif /* HISTORYSAMPLER_HPP_ */
