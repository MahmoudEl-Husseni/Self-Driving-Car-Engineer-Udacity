#include <iostream>
#include <algorithm>
#include <vector>

#include "help_functions.h"
using namespace std;

std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float position_stdev);

float motion_model(float pseudo_position, float movement, std::vector<float> priors,
                   int map_size, int control_stdev);

float observation_model(std::vector<float> landmark_positions, std::vector<float> observations,
                        std::vector<float> pseudo_ranges, float distance_max,
                        float observation_stdev);

std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, float pseudo_position);

int main()
{

    // set standard deviation of control:
    float control_stdev = 1.0f;

    // set standard deviation of position:
    float position_stdev = 1.0f;

    // meters vehicle moves per time step
    float movement_per_timestep = 1.0f;

    // number of x positions on map
    int map_size = 25;
    int distance_max = map_size;

    // initialize landmarks
    std::vector<float> landmark_positions{5, 10, 20};

    // define observations vector, each inner vector represents a set of observations
    // for a time step
    std::vector<std::vector<float>> sensor_obs{{1, 7, 12, 21}, {0, 6, 11, 20}, {5, 10, 19}, {4, 9, 18}, {3, 8, 17}, {2, 7, 16}, {1, 6, 15}, {0, 5, 14}, {4, 13}, {3, 12}, {2, 11}, {1, 10}, {0, 9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}, {0}, {}, {}, {}};
    // initialize priors
    std::vector<float> priors = initialize_priors(map_size, landmark_positions,
                                                  position_stdev);

    // initialize posteriors
    std::vector<float> posteriors(map_size, 0.0);

    vector<float> observations;
    for (int t = 0; t < sensor_obs.size(); ++t){
        if (!sensor_obs[t].empty())
        {
            observations = sensor_obs[t];
        }
        else
        {
            observations = {float(distance_max)};
        }

        for (unsigned int i = 0; i < map_size; ++i)
        {
            float pseudo_position = float(i);

            // TODO: get the motion model probability for each x position
            float motion_prob = motion_model(pseudo_position, movement_per_timestep, priors,
                                             map_size, control_stdev);
            // TODO: get pseudo ranges
            vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, pseudo_position);

            // TODO: get observation probability
            float observation_prob = observation_model(landmark_positions, observations,
                                                       pseudo_ranges, distance_max,
                                                       position_stdev);

            // TODO: calculate the ith posterior and pass to posteriors vector
            posteriors[i] = motion_prob * observation_prob;

            
            std::cout << motion_prob << "\t" << observation_prob << "\t"
            << "\t"  << motion_prob * observation_prob << endl;
        }

        // normalize posteriors
        posteriors = Helpers::normalize_vector(posteriors);
        
    }
    return 0;
};

// TODO, implement the motion model: calculates prob of being at an estimated position at time t
float motion_model(float pseudo_position, float movement, std::vector<float> priors,
                   int map_size, int control_stdev)
{

    // initialize probability
    float position_prob = 0.0f;

    // loop over state space for all possible positions x (convolution):
    for (auto prior : priors)
    {
        // estimate the motion for each x position
        float next_pseudo_position = fmod((pseudo_position + movement), map_size);
        // calculate the transition probability:
        float transition_prob = Helpers::normpdf(next_pseudo_position, pseudo_position, control_stdev);
        // estimate probability for the motion model, this is our prior
        position_prob += transition_prob * prior;
    }
    return position_prob;
}

// initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float position_stdev)
{
    // initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev

    // set all priors to 0.0
    std::vector<float> priors(map_size, 0.0);

    // set each landmark positon +/1 to 1.0/9.0 (9 possible postions)
    float normalization_term = landmark_positions.size() * (position_stdev * 2 + 1);
    for (unsigned int i = 0; i < landmark_positions.size(); i++)
    {
        int landmark_center = landmark_positions[i];
        priors[landmark_center] = 1.0f / normalization_term;
        priors[landmark_center - 1] = 1.0f / normalization_term;
        priors[landmark_center + 1] = 1.0f / normalization_term;
    }
    return priors;
}

float observation_model(std::vector<float> landmark_positions, std::vector<float> observations,
                        std::vector<float> pseudo_ranges, float distance_max,
                        float observation_stdev)
{

    float distance_prob = 1.0;
    for (unsigned int i = 0; i < landmark_positions.size(); ++i)
    {
        // estimate the probability for observation model, this is our likelihood
        for (int i = 0; i < landmark_positions.size(); i++)
        {
            int pseudo_range_min;
            if (pseudo_ranges.size() > 0)
            {
                // set min distance:
                pseudo_range_min = pseudo_ranges[0];
                // remove this entry from pseudo_ranges-vector:
                pseudo_ranges.erase(pseudo_ranges.begin());
            }
            else
            {
                pseudo_range_min = distance_max;
            }
            distance_prob *= Helpers::normpdf(observations[i], pseudo_range_min, observation_stdev);
        }
    }

    return distance_prob;
}

std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, float pseudo_position)
{

    // define pseudo observation vector:
    std::vector<float> pseudo_ranges;

    for (int i = 0; i < landmark_positions.size(); i++)
    {
        if (landmark_positions[i] > pseudo_position)
        {
            pseudo_ranges.push_back(landmark_positions[i] - pseudo_position);
        }
    }
    sort(pseudo_ranges.begin(), pseudo_ranges.end());
    return pseudo_ranges;
}