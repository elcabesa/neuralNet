#ifndef _LABELED_EXAMPLE_H
#define _LABELED_EXAMPLE_H

#include <vector>
#include "sparse.h"

class LabeledExample {
public:
    LabeledExample(const std::vector<double>& in, double l):features(in), label(l){};
    SparseInput features;
    double label;
};

#endif
