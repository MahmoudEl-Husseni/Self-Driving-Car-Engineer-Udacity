#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class utils
{
public:
    /**
     * Constructor.
     */
    utils();

    /**
     * Destructor.
     */
    virtual ~utils();

    VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                           const vector<VectorXd> &ground_truth);
    /**
     * A helper method to calculate Jacobians.
     */
    MatrixXd CalculateJacobian(const VectorXd &x_state);
};

#endif /* TOOLS_H_ */