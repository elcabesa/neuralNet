#ifndef _STATS_H
#define _STATS_H

#include <vector>

class Model;

class Stats {
public:
    Stats(Model& m);
    void update();
    void print() const;
private:
    const Model& _m;
    std::vector<std::vector<unsigned long long int>> _dyingReluCounter;
    unsigned long long int _examples;
};

#endif