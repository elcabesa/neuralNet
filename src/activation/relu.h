#ifndef _RELU_H
#define _RELU_H

#include "activation.h"

class reluActivation: public Activation{
public:
    reluActivation();
    ~reluActivation();
    
    double propagate(double input) const;
    double derivate(double input) const;
    type getType() const;
    static constexpr double alpha = 0.0001;
    static constexpr double saturation = 127.0;
}; 


#endif 
