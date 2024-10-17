#include <numeric>
#include <vector>

namespace linalg {

double dot(std::vector<double>& v1, std::vector<double>& v2) {
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

}  // namespace linalg
