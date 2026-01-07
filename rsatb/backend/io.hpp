#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <cmath>
#include <algorithm>

namespace rsatb {
namespace backend {
namespace io {

// Helper to split string
inline std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (tokenStream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Simple atomic number map
inline int get_atomic_number(const std::string& symbol) {
    static std::map<std::string, int> table = {
        {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18},
        {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
        {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36},
        {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48},
        {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54},
        {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71},
        {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80},
        {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86},
        {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90}, {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}, {"Fm", 100}
    };
    if (table.find(symbol) != table.end()) return table[symbol];
    try {
        return std::stoi(symbol);
    } catch (...) {
        return 0; 
    }
}

struct StructureData {
    std::vector<double> pos; // 3*N
    std::vector<int> z;      // N
    std::vector<double> cell; // 9
    std::vector<bool> pbc;    // 3
    int n_atom;
};

inline StructureData read_poscar(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
    
    StructureData data;
    std::string line;
    std::vector<std::string> tokens;
    
    // 1. Comment
    if (!std::getline(file, line)) throw std::runtime_error("Empty POSCAR");
    
    // 2. Scale
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (scale)");
    double scale = std::stod(line);
    
    // 3-5. Cell
    data.cell.resize(9);
    for(int i=0; i<3; ++i) {
        if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (cell)");
        tokens = split(line);
        if (tokens.size() < 3) throw std::runtime_error("Invalid cell line");
        for(int j=0; j<3; ++j) data.cell[3*i+j] = std::stod(tokens[j]) * scale;
    }
    data.pbc = {true, true, true};
    
    // 6. Elements (VASP 5) or Counts (VASP 4)
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (species/counts)");
    tokens = split(line);
    if (tokens.empty()) throw std::runtime_error("Empty species/counts line");
    
    std::vector<std::string> elements;
    std::vector<int> counts;
    
    bool vasp5 = false;
    if (std::isalpha(tokens[0][0])) {
        vasp5 = true;
        elements = tokens;
        if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (counts)");
        tokens = split(line);
    }
    
    for(const auto& t : tokens) counts.push_back(std::stoi(t));
    
    data.n_atom = 0;
    for(int c : counts) data.n_atom += c;
    
    data.pos.resize(data.n_atom * 3);
    data.z.resize(data.n_atom);
    
    // 7. Coordinate type
    if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (coord type)");
    bool direct = (line[0] == 'D' || line[0] == 'd'); // Direct or Cartesian
    
    // 8. Read positions
    int idx = 0;
    for(size_t i=0; i<counts.size(); ++i) {
        int z_val = 0;
        if (vasp5 && i < elements.size()) z_val = get_atomic_number(elements[i]);
        else z_val = i + 1; // Dummy Z if not VASP 5
        
        for(int j=0; j<counts[i]; ++j) {
            if (!std::getline(file, line)) throw std::runtime_error("Truncated POSCAR (positions)");
            tokens = split(line);
            if (tokens.size() < 3) throw std::runtime_error("Invalid position line");
            double x = std::stod(tokens[0]);
            double y = std::stod(tokens[1]);
            double z_coord = std::stod(tokens[2]);
            
            if (direct) {
                data.pos[3*idx] = x*data.cell[0] + y*data.cell[3] + z_coord*data.cell[6];
                data.pos[3*idx+1] = x*data.cell[1] + y*data.cell[4] + z_coord*data.cell[7];
                data.pos[3*idx+2] = x*data.cell[2] + y*data.cell[5] + z_coord*data.cell[8];
            } else {
                data.pos[3*idx] = x * scale;
                data.pos[3*idx+1] = y * scale;
                data.pos[3*idx+2] = z_coord * scale;
            }
            data.z[idx] = z_val;
            idx++;
        }
    }
    
    return data;
}

inline StructureData read_xyz(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
    
    StructureData data;
    std::string line;
    
    // 1. N atoms
    if (!std::getline(file, line)) throw std::runtime_error("Empty XYZ");
    try {
        data.n_atom = std::stoi(line);
    } catch (...) {
        throw std::runtime_error("Invalid XYZ header (N atoms)");
    }
    
    // 2. Comment (Lattice?)
    if (!std::getline(file, line)) throw std::runtime_error("Truncated XYZ (comment)");
    
    data.cell = {0,0,0, 0,0,0, 0,0,0};
    data.pbc = {false, false, false};
    
    size_t lat_pos = line.find("Lattice=\"");
    if (lat_pos != std::string::npos) {
        std::string lat_str = line.substr(lat_pos + 9);
        size_t end_pos = lat_str.find("\"");
        if (end_pos != std::string::npos) {
            lat_str = lat_str.substr(0, end_pos);
            std::vector<std::string> tokens = split(lat_str);
            if (tokens.size() >= 9) {
                for(int i=0; i<9; ++i) data.cell[i] = std::stod(tokens[i]);
                data.pbc = {true, true, true};
            }
        }
    }
    
    data.pos.resize(data.n_atom * 3);
    data.z.resize(data.n_atom);
    
    for(int i=0; i<data.n_atom; ++i) {
        if (!std::getline(file, line)) throw std::runtime_error("Truncated XYZ (positions)");
        std::vector<std::string> tokens = split(line);
        if (tokens.size() < 4) continue;
        
        data.z[i] = get_atomic_number(tokens[0]);
        data.pos[3*i] = std::stod(tokens[1]);
        data.pos[3*i+1] = std::stod(tokens[2]);
        data.pos[3*i+2] = std::stod(tokens[3]);
    }
    
    return data;
}

} // namespace io
} // namespace backend
} // namespace rsatb
