p_init = [0.2, 0.2, 0.2, 0.2, 0.2]
world = ['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green'] # measurements after two moves
motions = [1, 1] # two moves of one step each to the right
p = []

pHit = 0.6 # probability of hitting the right color
pMiss = 0.2 # probability of missing the right color

pExact = 0.8 # probability of moving exactly
pOvershoot = 0.1 # probability of moving one step too far
pUndershoot = 0.1 # probability of moving one step too short


def sense(p, Z):
    '''
    calculate the probability of the robot being at a given position according to the measurement

    Args: 
        p: initial probability
        Z: measurement

    Returns: 
        q: probability of the robot being at a given position
    '''

    q = []
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
    '''
    calculate the probability of the robot being at a given position according to the movement
    Args: 
        p: initial probability
        U: movement
    Returns: 
        q: probability of the robot being at a given position
    '''
    q = []
    for i in range(len(p)):
        # after each move, the probability of the robot being at a given position is
        # the sum of the probability of the robot being at the previous position
        # and the probability of the robot being at the previous position plus one step.
        
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)
    return q

for k in range(len(measurements)):
    p = sense(p_init, measurements[k])
    p = move(p, motions[k])

print(p)
