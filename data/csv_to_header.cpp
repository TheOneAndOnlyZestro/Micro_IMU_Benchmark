#include "csv_to_header.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

void write_csv_to_header_file(const char* csv_filepath, const char* destination_filepath)
{
    std::ifstream csvInput(csv_filepath);
    if (!csvInput.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csv_filepath << std::endl;
        return;
    }

    std::string line;
    std::vector<std::vector<float>> data;

    // 1. Skip the header line
    if (!std::getline(csvInput, line)) {
        std::cerr << "Error: CSV file is empty" << std::endl;
        return;
    }

    // 2. Read data rows
    while (std::getline(csvInput, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Skipping invalid float value: " << cell << std::endl;
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    csvInput.close();

    if (data.empty()) {
        std::cerr << "Error: No valid data found in CSV" << std::endl;
        return;
    }

    // 3. Write to header file
    std::ofstream headerOutput(destination_filepath);
    if (!headerOutput.is_open()) {
        std::cerr << "Error: Could not create header file: " << destination_filepath << std::endl;
        return;
    }

    size_t rows = data.size();
    size_t cols = data[0].size();

    // Write Header Guards and Dimensions
    headerOutput << "#ifndef CSV_DATA_H\n";
    headerOutput << "#define CSV_DATA_H\n\n";
    headerOutput << "const int CSV_ROWS = " << rows << ";\n";
    headerOutput << "const int CSV_COLS = " << cols << ";\n\n";

    // Write the 2D Array
    headerOutput << "const float csv_data[" << rows << "][" << cols << "] = {\n";
    for (size_t i = 0; i < rows; ++i) {
        headerOutput << "    { ";
        for (size_t j = 0; j < cols; ++j) {
            headerOutput << data[i][j] << (j == cols - 1 ? "" : ", ");
        }
        headerOutput << " }" << (i == rows - 1 ? "" : ",") << "\n";
    }
    headerOutput << "};\n\n";

    headerOutput << "#endif // CSV_DATA_H\n";
    headerOutput.close();

    std::cout << "Successfully wrote " << rows << "x" << cols << " array to " << destination_filepath << std::endl;
}
