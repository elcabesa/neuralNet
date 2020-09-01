#ifndef _INPUT_H
#define _INPUT_H

#include <vector>

class Input {
public:
    Input(const std::vector<double> v);
    Input(const unsigned int size);
    ~Input();
    void print() const;
    const double& get(unsigned int index) const;
    double& get(unsigned int index);
    unsigned int getElementNumber() const;
    double& getElementFromIndex( unsigned int index);
    const double& getElementFromIndex(unsigned int index) const;
    unsigned int getPositionFromIndex(unsigned int index) const;
private:
    unsigned int _size;
    std::vector<double> _in;
};

#endif  
