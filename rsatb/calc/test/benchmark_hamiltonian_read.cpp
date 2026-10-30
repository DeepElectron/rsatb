// Distributed Hamiltonian-file read: correctness against tiling, and cost.
//
// A primitive-cell H(R) tiled L x L x L by the structure is, block for block,
// the H(R) of the L^3 cell written out whole. So a big-cell file produced by
// folding the primitive one (make_big_hamiltonian_files.py, alongside) must
// give exactly the images the primitive file gives through supercell tiling,
// on every rank of any partition. This driver builds the partitioned graph
// the way RSATB does, reads the primitive DeePTB file tiled as the reference,
// then reads each big file untiled -- the regime where a reader has to be
// distributed -- and reports per-read wall time (max over ranks), peak RSS
// (max over ranks) and the largest block difference against the reference.
//
// Usage:
//   benchmark_hamiltonian_read --poscar conf.vasp --primitive hamiltonians.h5
//       --supercell L --cutoff 3.7042 --norb 21 --shells 0,0,0,0,1,1,1,1,2
//       [--abacus big.csr]... [--nextham HR_atom_pairs.h5]...

#include "../../model/hamiltonian_images.hpp"
#include "../../io/hamiltonian_readers.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/io.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using vbcsr::atomic::AtomicData;
using Images = vbcsr::atomic::ImageContainer<double>;

// Peak resident set since the last ResetPeak, in MB (VmHWM; see clear_refs).
double PeakMb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
    }
    return -1.0;
}

void ResetPeak() {
    std::ofstream("/proc/self/clear_refs") << "5";
}

double MaxOverRanks(double value) {
    double out = value;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
}

// max |A(k) - B(k)| over this rank's blocks, at a generic k.
double MaxDifference(Images& a, Images& b) {
    const std::vector<double> k{0.11, 0.23, 0.37};
    std::unique_ptr<vbcsr::BlockSpMat<std::complex<double>>> ma(a.sample_k(k, vbcsr::atomic::PhaseConvention::R_ONLY));
    std::unique_ptr<vbcsr::BlockSpMat<std::complex<double>>> mb(b.sample_k(k, vbcsr::atomic::PhaseConvention::R_ONLY));
    const auto* g = ma->graph;
    double worst = 0.0;
    const int n_rows = static_cast<int>(g->owned_global_indices.size());
    for (int i = 0; i < n_rows; ++i) {
        for (int e = g->adj_ptr[i]; e < g->adj_ptr[i + 1]; ++e) {
            const size_t size = static_cast<size_t>(g->block_sizes[i]) * g->block_sizes[g->adj_ind[e]];
            const auto* x = ma->block_data(e);
            const auto* y = mb->block_data(e);
            for (size_t v = 0; v < size; ++v) worst = std::max(worst, std::abs(x[v] - y[v]));
        }
    }
    return worst;
}

std::vector<int> ParseInts(const std::string& text) {
    std::vector<int> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) out.push_back(std::stoi(item));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 0;
    try {
        std::string poscar, primitive;
        int repeat = 1, norb = 0;
        double cutoff = 0.0;
        std::vector<int> shells;
        std::vector<std::pair<std::string, rsatb::io::HamiltonianFormat>> files;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            auto next = [&]() {
                if (i + 1 >= argc) throw std::runtime_error("missing value after " + flag);
                return std::string(argv[++i]);
            };
            if (flag == "--poscar") poscar = next();
            else if (flag == "--primitive") primitive = next();
            else if (flag == "--supercell") repeat = std::stoi(next());
            else if (flag == "--cutoff") cutoff = std::stod(next());
            else if (flag == "--norb") norb = std::stoi(next());
            else if (flag == "--shells") shells = ParseInts(next());
            else if (flag == "--abacus") files.emplace_back(next(), rsatb::io::HamiltonianFormat::kAbacusCsr);
            else if (flag == "--nextham") files.emplace_back(next(), rsatb::io::HamiltonianFormat::kNextHamHdf5);
            else throw std::runtime_error("unknown argument " + flag);
        }

        auto st = vbcsr::atomic::ReadStructure(poscar);
        const int primitive_n_atom = st.n_atom;
        st.supercell = {{repeat, repeat, repeat}};
        const auto expanded = st.BuildSupercell();
        std::unique_ptr<AtomicData> ad(AtomicData::from_points(
            expanded.pos, expanded.z, expanded.cell, expanded.pbc, {cutoff}, {norb}, MPI_COMM_WORLD));
        ad->primitive_n_atom = primitive_n_atom;
        ad->structure_supercell = {{repeat, repeat, repeat}};
        if (rank == 0) {
            std::printf("%d ranks, %d atoms (%d^3 x %d), %d orbitals/atom\n", size, ad->N_atom, repeat,
                        primitive_n_atom, norb);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        auto deeptb = rsatb::io::OpenHamiltonianFile(
            rsatb::io::HamiltonianFile{primitive, rsatb::io::HamiltonianFormat::kDeeptbHdf5});
        auto reference = rsatb::model::BuildHamiltonianImagesFromFile(*ad, *deeptb);
        const double t_ref = MaxOverRanks(MPI_Wtime() - t0);
        if (rank == 0) std::printf("reference  deeptb primitive, tiled          read %8.3f s\n", t_ref);

        // From here on the file is the whole cell, numbered as the expansion
        // numbers its atoms: no folding.
        ad->primitive_n_atom = ad->N_atom;
        ad->structure_supercell = {{1, 1, 1}};
        rsatb::io::FileBasisLayout layout;
        layout.type_shells = {shells};
        layout.type_atomic_number = {expanded.z.front()};
        layout.atom_type.assign(static_cast<size_t>(ad->N_atom), 0);

        for (const auto& file : files) {
            ResetPeak();
            const double base = PeakMb();
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            auto reader = rsatb::io::OpenHamiltonianFile(
                rsatb::io::HamiltonianFile{file.first, file.second}, layout, MPI_COMM_WORLD);
            const double t_open = MaxOverRanks(MPI_Wtime() - t0);
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            auto images = rsatb::model::BuildHamiltonianImagesFromFile(*ad, *reader);
            const double t_read = MaxOverRanks(MPI_Wtime() - t0);
            const double peak = MaxOverRanks(PeakMb() - base);
            const double least = -MaxOverRanks(-(PeakMb() - base));
            const double diff = MaxOverRanks(MaxDifference(*reference, *images));
            if (rank == 0) {
                const std::string name = file.first.substr(file.first.find_last_of('/') + 1);
                std::printf("%-7s %-26s open %6.3f s  read %7.3f s  peak +%5.0f..%5.0f MB/rank  max|dH| %.2e\n",
                            rsatb::io::ToString(file.second).c_str(), name.c_str(), t_open, t_read,
                            least, peak, diff);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
        status = 1;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
