// Reference runner for rcad-mbd Tier B: for each (input, output) pair,
// parse the .asmt, run the kinematic analysis, write the result with its
// time series.
#include <iostream>
#include <exception>
#include "ASMTAssembly.h"

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2 != 0) {
        std::cerr << "usage: ondsel_run in.asmt out.asmt [in2 out2 ...]\n";
        return 2;
    }
    int failures = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        try {
            auto assembly = MbD::ASMTAssembly::assemblyFromFile(argv[i]);
            assembly->runKINEMATIC();
            assembly->outputFile(argv[i + 1]);
            std::cerr << "ok " << argv[i] << "\n";
        } catch (const std::exception& e) {
            std::cerr << "FAILED " << argv[i] << ": " << e.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
