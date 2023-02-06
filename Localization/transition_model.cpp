#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

// initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float position_stdev);

int main()
{

    // set standard deviation of position:
    float position_stdev = 1.0;

    // set map horizon distance in meters
    int map_size = 25;

    // initialize landmarks
    std::vector<float> landmark_positions{5, 10, 20};

    // initialize priors
    std::vector<float> priors = initialize_priors(map_size, landmark_positions,
                                                  position_stdev);

    // print values to stdout
    for (unsigned int p = 0; p < priors.size(); p++)
    {
        std::cout << p << ": " << priors[p] << endl;
    }

    return 0;
};

std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float position_stdev)
{

    // initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev

    // set all priors to 0.0
    std::vector<float> priors(map_size, 0.0);

    int sum = 0;
    for(auto &landmark : landmark_positions)
    {
        for (int i=-position_stdev; i<=position_stdev; i++)
        {
            priors[int(landmark + i + map_size)%map_size]++;
            sum++;
        }
    }
    float summ=0;
    for(auto &prior : priors)
    {
        prior /= sum;
        summ += prior;
    }
    cout << "sum: " << sum << endl;
    cout << "summ: " << summ << endl;

    return priors;
}
