#ifndef RSATB_TEST_DATA_PATH_HPP
#define RSATB_TEST_DATA_PATH_HPP

#include <cstdlib>
#include <string>

// Absolute location of <repo>/test/data, baked in by CMake so the test binaries
// do not depend on where they are run from.
#ifndef RSATB_TEST_DATA_DIR
#define RSATB_TEST_DATA_DIR ""
#endif

namespace rsatb {
namespace test {

// Root of the test data tree, always with a trailing '/'. The compiled-in source
// location can be overridden at runtime via the RSATB_TEST_DATA_DIR environment
// variable, for running the binaries against a relocated data tree.
inline std::string data_dir() {
    std::string dir;
    if (const char* env = std::getenv("RSATB_TEST_DATA_DIR")) {
        dir = env;
    }
    if (dir.empty()) {
        dir = RSATB_TEST_DATA_DIR;
    }
    if (!dir.empty() && dir.back() != '/') {
        dir.push_back('/');
    }
    return dir;
}

// Resolve a path relative to the test data root, e.g. data_path("diamond/POSCAR").
inline std::string data_path(const std::string& relative) {
    return data_dir() + relative;
}

} // namespace test
} // namespace rsatb

#endif // RSATB_TEST_DATA_PATH_HPP
