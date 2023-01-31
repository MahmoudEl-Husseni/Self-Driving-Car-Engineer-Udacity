from matrix import matrix

x = matrix([[0.], [0.]])                # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]])  # initial uncertainty
u = matrix([[0.], [0.]])                # external motion
F = matrix([[1., 1.], [0, 1.]])         # next state function
H = matrix([[1., 0.]])                  # measurement function
R = matrix([[1.]])                      # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]])        # identity matrix
measurements = [1, 2, 3]


def filter(X, P):

    for n in range(len(measurements)):
        # First: predict the current state and covariance using previous state and covariance
        # prediction
        
        X = (F * X) + u
        P = F * P * F.transpose()

        # Second: update the current state and covariance using the measurement and the predicted state and covariance from step 1
        Z = matrix([[measurements[n]]])
        Y = Z - (H * X)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        X = X + (K * Y)
        P = (I - (K * H)) * P

    return X, P

X, P = filter(x, P)
print('X = '), X.show()
print('P = '), P.show()
