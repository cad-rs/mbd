// Reference runner for rcad-mbd Tier B: for each (input, output) pair,
// parse the .asmt, run the kinematic analysis, write the result with its
// time series.
//
// The file is read here rather than through ASMTAssembly::assemblyFromFile:
// that function pops the "Assembly" header inside an assert(), so a release
// (NDEBUG) build skips the pop and the parse runs one line off. Lines are
// also stripped of a trailing '\r' (CRLF checkouts on Windows).
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ASMTAssembly.h"

static std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream stream(path);
    if (stream.fail()) {
        throw std::invalid_argument("file not found: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

static void run(const std::string& in, const std::string& out) {
    auto lines = read_lines(in);
    if (lines.size() < 2 ||
        (lines[0] != "OndselSolver" &&
         lines[0] != "freeCAD: 3D CAD with Motion Simulation  by  askoh.com") ||
        lines[1] != "Assembly") {
        throw std::invalid_argument("not an .asmt assembly: " + in);
    }
    lines.erase(lines.begin(), lines.begin() + 2);
    auto assembly = MbD::ASMTAssembly::With();
    assembly->setFilename(in);
    assembly->parseASMT(lines);
    assembly->runKINEMATIC();
    assembly->outputFile(out);
}

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2 != 0) {
        std::cerr << "usage: ondsel_run in.asmt out.asmt [in2 out2 ...]\n";
        return 2;
    }
    int failures = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        try {
            run(argv[i], argv[i + 1]);
            std::cerr << "ok " << argv[i] << "\n";
        } catch (const std::exception& e) {
            std::cerr << "FAILED " << argv[i] << ": " << e.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
