#include "Simulator/Simulator.h"

int main() {
    try {
        Matrix matrix;
        History hist;
        Result result;
        Simulator sim(matrix, hist, result);
        return 0;
    }

    catch (const std::exception &err) {
        if (strcmp(err.what(), "stoi") == 0)
            std::cerr << "Bad argument";
        else
            std::cerr << err.what();
        return 1;
    }
}