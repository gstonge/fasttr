#ifndef HISTORYSAMPLER_HPP_
#define HISTORYSAMPLER_HPP_

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

const double INF = std::numeric_limits<double>::infinity();

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
            const std::vector<double>& kernel_vector,
            const std::vector<double>& grad_kernel_vector,
            unsigned int seed = 42);

    //Accessors
    const AdjacencyMap& get_adjacency() const
        {return adjacency_map_;}
    const AdjacencyMap& get_rooted_adjacency() const
        {return rooted_adjacency_map_;}
    const std::unordered_map<Node,double>& get_probability() const
        {return probability_map_;}
    const std::vector<History>& get_histories() const
        {return history_vector_;}
    const std::vector<double>& get_log_posterior() const
        {return log_posterior_vector_;}
    const std::vector<double>& get_grad_log_posterior() const
        {return grad_log_posterior_vector_;}
    double get_ground_truth_log_posterior();
    std::unordered_map<Node,double> get_marginal_mean();

    //Mutators
    void root(const Node& source);
    void unroot();

    void sample(std::size_t nb_sample);
    void set_kernel_vector(const std::vector<double>& kernel_vector,
            const std::vector<double>& grad_kernel_vector)
        {kernel_vector_ = kernel_vector;
         grad_kernel_vector_ = grad_kernel_vector;
         update_log_posterior();}
    void set_ground_truth(const History& ground_truth)
        {ground_truth_ = ground_truth;}


private:
    AdjacencyMap adjacency_map_;
    AdjacencyMap rooted_adjacency_map_;
    std::vector<double> kernel_vector_;
    std::vector<double> grad_kernel_vector_;
    std::size_t n_;
    mutable sset::SamplableSetCR<Node> boundary_;
    mutable sset::SamplableSetCR<Node> source_;
    Node root_;
    bool rooted_;
    std::unordered_map<Node,
        std::unordered_map<Node,std::size_t>> descendant_map_;
    std::unordered_map<Node,double> probability_map_; //for source
    std::vector<History> history_vector_;
    std::vector<double> log_posterior_vector_; //for history
    std::vector<double> grad_log_posterior_vector_; //for history
    History ground_truth_;

    //methods
    std::size_t compute_descendant(const Node& node);
    void compute_source_probability(const Node& node);

    std::pair<double,double> compute_log_posterior(const History& history);
    void update_log_posterior();
};

//Default constructor
template <typename Node>
HistorySampler<Node>::HistorySampler(
        const std::unordered_map<Node,std::unordered_set<Node>>& adjacency_map,
        const std::vector<double>& kernel_vector,
        const std::vector<double>& grad_kernel_vector,
        unsigned int seed) :
    adjacency_map_(adjacency_map),
    rooted_adjacency_map_(),
    kernel_vector_(kernel_vector),
    grad_kernel_vector_(grad_kernel_vector),
    n_(adjacency_map_.size()),
    boundary_(1,n_,seed),
    source_(1,1,seed+1),
    root_(),
    rooted_(false),
    descendant_map_(),
    probability_map_(),
    history_vector_(),
    log_posterior_vector_(),
    ground_truth_()
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

    //normalize and determine min max prob
    double weight_sum = 0.;
    for (const auto& element : probability_map_)
    {
        weight_sum += element.second;
    }
    double min_probability = 1.;
    double max_probability = 0.;
    double p;
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
    source_ = sset::SamplableSetCR<Node>(min_probability,max_probability,seed+1);
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

//sample a certain number of histories
template <typename Node>
void HistorySampler<Node>::sample(std::size_t nb_sample)
{
    history_vector_.clear();
    for (std::size_t i = 0; i < nb_sample; i++)
    {
        history_vector_.push_back(History());
        History& history = history_vector_[i];
        Node source = (source_.sample()).first;
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
            boundary_.insert(neighbor, descendant_[neighbor]);
        }
        while (boundary_.size() > 0)
        {
            Node node = (boundary_.sample()).first;
            boundary_.erase(node);
            history.push_back(node);
            //add neighbors of node to boundary
            for (const auto& neighbor : rooted_adjacency_map_[node])
            {
                boundary_.insert(neighbor, descendant_[neighbor]);
            }
        }
        unroot();
    }
    //get the log probability associated with each history
    update_log_posterior();
}

//get log probability of an history, and the gradient of that according to
//some parameter
template <typename Node>
std::pair<double,double> HistorySampler<Node>::compute_log_posterior(
        const History& history)
{
    std::unordered_map<Node,std::size_t> degree_map;
    degree_map[history[0]] = 1;
    degree_map[history[1]] = 1;
    double Z = 2*kernel_vector_[1];
    double dZ = 2*grad_kernel_vector_[1];
    double log_prob = 0.;
    double grad_log_prob = 0.;
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
    log_posterior_vector_.clear();
    log_posterior_vector_.reserve(history_vector_.size());
    //get log probability
    for (const auto& history : history_vector_)
    {
        std::pair<double,double> grad_log_prob_pair = compute_log_posterior(history);
        log_posterior_vector_.push_back(grad_log_prob_pair.first);
        grad_log_posterior_vector_.push_back(grad_log_prob_pair.second);
    }
}


//return a map of the marginal mean arrival time for each node
template <typename Node>
std::unordered_map<Node,double> HistorySampler<Node>::get_marginal_mean()
{
    std::unordered_map<Node,double> marginal_mean;
    double max_log_prob = *std::max_element(log_posterior_vector_.begin(),
            log_posterior_vector_.end());
    std::vector<double> probability_vector(log_posterior_vector_);
    //get weight
    double total_prob = 0;
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
double HistorySampler<Node>::get_ground_truth_log_posterior()
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
