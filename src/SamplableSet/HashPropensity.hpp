/*
 * MIT License
 *
 * Copyright (c) 2018 Guillaume St-Onge
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HASHPROPENSITY_HPP_
#define HASHPROPENSITY_HPP_

#include <cstdlib>

namespace sset
{//start of namespace sset


// Unary function object to hash the propensity of events to groups
class HashPropensity
{
public:
    //Constructor
    HashPropensity(long double propensity_min, long double propensity_max);
    HashPropensity(const HashPropensity& hash_object);

    //Call operator definition
    std::size_t operator()(long double propensity) const;

private:
    //Members
    long double propensity_min_;
    long double propensity_max_;
    bool power_of_two_;
};

}//end of namespace sset

#endif /* HASHPROPENSITY_HPP_ */
