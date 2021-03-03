#include <iostream>
#include <string>
#include <random>
#include <vector>

using namespace std;

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end);
  return linspaced;
}

int main() {

    // system configuration parameters
    auto num = 10;
    auto vol = 8.0;
    auto fvol2 = 5.0;
    auto temp = 0.5;
    auto dt = 1e-3;
    auto visc = 5.0;
    auto step_max = 100000;
    auto dim = 2;
    auto write_step = 10;
    auto stdout_step = 10000;
    auto seed = 1; // TODO use random seed
    
    // generate simultion box from config params
    auto sigma = 1.0;
    auto fsigma2 = 1.0;
    auto beta = 1.0/temp;

    double b[3] = {1.0, 1.0, 1.0};
    double bh[3] = {.5, .5, .5};
    auto l = pow(vol, 1/((double) dim));
    auto l2 = l/2.0;

    for (int i=0; i < dim; i++) {
        b[i] = l;
        bh[i] = l2;
    }

    std::default_random_engine generator;

    std::uniform_real_distribution<double> uniform_distribution(0.0,1.0);
    std::normal_distribution<double> normal_distribution(0.0,sqrt(dt));

    vector<double [3]> x{};

    if (dim == 3) {
        fsigma2 = cbrt(vol/fvol2);
        for (int i=0; i < num; i++) {
            x.push_back({uniform_distribution(generator)*l - l2, 
                    uniform_distribution(generator)*l - l2, 
                    uniform_distribution(generator)*l - l2});
        }
    } else if (dim == 2) {
        fsigma2 = sqrt(vol/fvol2);
        for (int i=0; i < num; i++) {
            x.push_back({uniform_distribution(generator)*l - l2, 
                    uniform_distribution(generator)*l - l2, 
                    0.0});
        }
    }
    else {
        return 1;
    }

    // these terms will be used a lot
    auto a_term = dt/visc;
    auto b_term = sqrt(2.0/(visc*beta));

    auto integration_factor = 0.0;
    auto step = 0;

    double w[dim*num] = {0};

    // a tad memory wasteful using linspace not as a generator
    for (auto&& sigma_b: linspace(sigma, fsigma2, step_max)) {

        // get w
        for (int i=0; i < dim*num; i++) {
            w[i] = normal_distribution(generator);
        }

        // get forces a
        
        // get forces b

        // calculate force bias

        // add to integration factor

        // dump output

    }

    return 0;
}

