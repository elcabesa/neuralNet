#ifndef _COST_H
#define _COST_H


class Cost {
public:
    Cost();
    ~Cost();    
    
    double calc(const double out, const double label) const;
    double derivate(const double out, const double label) const;
private:
    double sigmoid(double x) const;
    static constexpr double scaleFactor = 0.000333333;
    static constexpr double multiplier = 12000;
};

#endif  
