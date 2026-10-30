// Readers for ABACUS CSR and NextHam HDF5 Hamiltonians.
//
// One synthetic H(R), in ABACUS's orbital order and in Rydberg, is written in
// every layout these readers accept -- ABACUS text, ABACUS binary (with the
// older zero-row-offset record for an empty R), and NextHam HDF5 in eV -- and
// each reader must return the same blocks: ABACUS's, put into rsatb's orbital
// order and converted to eV. The orbital map itself is pinned against the
// convention written out by hand.
//
// Run it on several ranks too (mpirun -np 3): each rank asks for a different
// share of the blocks, as the image builder does, and must get exactly its
// own back. That is what exercises the text reader's scatter from rank 0 and
// the binary and NextHam readers' per-rank row reads.

#include "hamiltonian_readers.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using rsatb::io::AbacusOrbitalOrder;
using rsatb::io::BlockKey;
using rsatb::io::DenseBlock;
using rsatb::io::FileBasisLayout;
using rsatb::io::HamiltonianFile;
using rsatb::io::HamiltonianFormat;
using Shift = std::array<int, 3>;

const std::string kText = "/tmp/rsatb_test_abacus_text.csr";
const std::string kBinary = "/tmp/rsatb_test_abacus_binary.csr";
const std::string kNextHam = "/tmp/rsatb_test_nextham.h5";

int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int Size() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

// Two species, an s+p atom (4 orbitals, Z = 6) and an s+p+d atom (9, Z = 14),
// as atoms C Si Si C: a 26-orbital basis.
FileBasisLayout FourAtomLayout() {
    FileBasisLayout layout;
    layout.type_shells = {{0, 1}, {0, 1, 2}};
    layout.type_atomic_number = {6, 14};
    layout.atom_type = {0, 1, 1, 0};
    return layout;
}

// The synthetic H(R): random ABACUS-order matrices over the 26-orbital basis,
// in Ry, with a third of the entries zero so rows are genuinely sparse.
// R = (0, 0, 1) is present but empty, and R = (0, -1, 1) couples only atom 0
// to atoms 1 and 2.
struct Fixture {
    int n = 26;
    std::array<int, 5> offsets{{0, 4, 13, 22, 26}};
    std::vector<Shift> shifts = {{{0, 0, 0}}, {{1, 0, 0}}, {{-1, 0, 0}}, {{0, 0, 1}}, {{0, -1, 1}}};
    std::map<Shift, std::vector<double>> dense;  // row-major n x n

    Fixture() {
        std::mt19937 engine(7);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        for (const auto& r : shifts) {
            std::vector<double> m(static_cast<size_t>(n * n), 0.0);
            if (r != Shift{{0, 0, 1}}) {
                for (auto& x : m) {
                    const double v = value(engine);
                    x = std::abs(v) < 0.33 ? 0.0 : v;
                }
                if (r == Shift{{0, -1, 1}}) {
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            if (!(a < 4 && b >= 4 && b < 22)) m[a * n + b] = 0.0;
                }
            }
            dense[r] = m;
        }
    }

    int Nnz(const Shift& r) const {
        int count = 0;
        for (double x : dense.at(r)) count += x != 0.0;
        return count;
    }

    // The ABACUS-order block of atoms (i, j), or empty if it is all zero.
    std::vector<double> AbacusBlock(int i, int j, const Shift& r) const {
        std::vector<double> block;
        bool any = false;
        for (int a = offsets[i]; a < offsets[i + 1]; ++a)
            for (int b = offsets[j]; b < offsets[j + 1]; ++b) {
                block.push_back(dense.at(r)[a * n + b]);
                any = any || block.back() != 0.0;
            }
        return any ? block : std::vector<double>{};
    }

    // Every (i, j, R) of the fixture, plus one the file cannot hold.
    std::vector<BlockKey> AllKeys() const {
        std::vector<BlockKey> keys;
        for (const auto& r : shifts)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) keys.push_back(BlockKey{i, j, r});
        keys.push_back(BlockKey{0, 1, {{5, 5, 5}}});
        return keys;
    }
};

void WriteText(const Fixture& f) {
    std::ofstream out(kText);
    out << "STEP: 0\nMatrix Dimension of H(R): " << f.n << "\nMatrix number of H(R): "
        << f.shifts.size() << "\n";
    out.precision(17);
    for (const auto& r : f.shifts) {
        out << r[0] << " " << r[1] << " " << r[2] << " " << f.Nnz(r) << "\n";
        if (f.Nnz(r) == 0) continue;  // current ABACUS writes nothing more
        const auto& m = f.dense.at(r);
        std::vector<int> offsets{0};
        for (int a = 0; a < f.n * f.n; ++a)
            if (m[a] != 0.0) out << " " << m[a];
        out << "\n";
        for (int a = 0; a < f.n; ++a) {
            int row = 0;
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != 0.0) {
                    out << " " << b;
                    ++row;
                }
            offsets.push_back(offsets.back() + row);
        }
        out << "\n";
        for (int x : offsets) out << " " << x;
        out << "\n";
    }
}

void WriteBinary(const Fixture& f) {
    std::ofstream out(kBinary, std::ios::binary);
    auto put = [&](std::int32_t x) { out.write(reinterpret_cast<const char*>(&x), 4); };
    put(0);
    put(f.n);
    put(static_cast<std::int32_t>(f.shifts.size()));
    for (const auto& r : f.shifts) {
        put(r[0]); put(r[1]); put(r[2]); put(f.Nnz(r));
        const auto& m = f.dense.at(r);
        if (f.Nnz(r) == 0) {
            // The older layout: an all-zero row-offset array for an empty R.
            for (int a = 0; a <= f.n; ++a) put(0);
            continue;
        }
        for (double x : m)
            if (x != 0.0) out.write(reinterpret_cast<const char*>(&x), 8);
        for (int a = 0; a < f.n; ++a)
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != 0.0) put(b);
        std::int32_t total = 0;
        put(0);
        for (int a = 0; a < f.n; ++a) {
            for (int b = 0; b < f.n; ++b) total += m[a * f.n + b] != 0.0;
            put(total);
        }
    }
}

void WriteStringAttribute(H5::H5File& file, const std::string& name, const std::string& value) {
    H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    file.createAttribute(name, type, H5::DataSpace(H5S_SCALAR)).write(type, value);
}

void WriteIntegerAttribute(H5::H5File& file, const std::string& name, long long value) {
    file.createAttribute(name, H5::PredType::STD_I64LE, H5::DataSpace(H5S_SCALAR))
        .write(H5::PredType::NATIVE_LLONG, &value);
}

void WriteInt64(H5::H5File& file, const std::string& name, const std::vector<std::int64_t>& v,
                hsize_t width) {
    hsize_t dims[2] = {v.size() / width, width};
    H5::DataSpace space(width == 1 ? 1 : 2, dims);
    file.createDataSet(name, H5::PredType::STD_I64LE, space).write(v.data(), H5::PredType::NATIVE_INT64);
}

// Writes the fixture the way NextHam's write_pair_hdf5 does, in eV.
void WriteNextHam(const Fixture& f, const std::string& op, long long nspin) {
    H5::H5File file(kNextHam, H5F_ACC_TRUNC);
    WriteStringAttribute(file, "format", "abacus_atom_pair_operator_v1");
    WriteStringAttribute(file, "operator", op);
    WriteStringAttribute(file, "HR_unit", "eV");
    WriteIntegerAttribute(file, "nspin", nspin);
    WriteIntegerAttribute(file, "basis_num", f.n);
    WriteIntegerAttribute(file, "atom_count", 4);

    std::vector<std::int64_t> pairs, pair_offset{0}, shifts, data_offset{0}, atom_pair_offset{0};
    std::vector<double> data;  // interleaved re, im
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::vector<Shift> present;
            for (const auto& r : f.shifts)
                if (!f.AbacusBlock(i, j, r).empty()) present.push_back(r);
            std::sort(present.begin(), present.end());
            if (present.empty()) continue;
            pairs.push_back(i);
            pairs.push_back(j);
            for (const auto& r : present) {
                shifts.insert(shifts.end(), {r[0], r[1], r[2]});
                for (double x : f.AbacusBlock(i, j, r)) {
                    data.push_back(x * rsatb::io::kAbacusEvPerRydberg);
                    data.push_back(0.0);
                }
            }
            pair_offset.push_back(static_cast<std::int64_t>(shifts.size() / 3));
            data_offset.push_back(static_cast<std::int64_t>(data.size() / 2));
        }
        atom_pair_offset.push_back(static_cast<std::int64_t>(pairs.size() / 2));
    }
    WriteInt64(file, "atom_basis_count", {4, 9, 9, 4}, 1);
    WriteInt64(file, "atom_basis_start", {0, 4, 13, 22}, 1);
    WriteInt64(file, "atomic_numbers", {6, 14, 14, 6}, 1);
    WriteInt64(file, "atom_pair_offset", atom_pair_offset, 1);
    WriteInt64(file, "pair_indices", pairs, 2);
    WriteInt64(file, "pair_offset", pair_offset, 1);
    WriteInt64(file, "R", shifts, 3);
    WriteInt64(file, "data_offset", data_offset, 1);

    H5::CompType complex_type(static_cast<size_t>(16));
    complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
    complex_type.insertMember("i", 8, H5::PredType::NATIVE_DOUBLE);
    hsize_t n_value = data.size() / 2;
    file.createDataSet("data", complex_type, H5::DataSpace(1, &n_value)).write(data.data(), complex_type);
}

// Rank 0 writes, everyone waits: the files are shared by all ranks.
template <typename Write>
void OnRootThenBarrier(Write write) {
    if (Rank() == 0) write();
    MPI_Barrier(MPI_COMM_WORLD);
}

// This rank's share of `keys`, round-robin, as a partitioned graph would ask.
std::vector<BlockKey> ShareOf(const std::vector<BlockKey>& keys) {
    std::vector<BlockKey> mine;
    for (size_t k = 0; k < keys.size(); ++k)
        if (static_cast<int>(k % Size()) == Rank()) mine.push_back(keys[k]);
    return mine;
}

// Reads `requests`, and checks every answer against ABACUS's block in rsatb order.
void ExpectFixtureBlocks(const rsatb::io::HamiltonianReader& reader, const Fixture& f,
                         const std::vector<BlockKey>& requests) {
    const auto layout = FourAtomLayout();
    const auto orders = rsatb::io::MakeAbacusOrbitalOrders(layout);
    std::vector<int> answered(requests.size(), 0);
    reader.ReadBlocks(requests, [&](size_t k, const DenseBlock& got) {
        ++answered[k];
        const BlockKey& key = requests[k];
        const auto abacus = f.AbacusBlock(key.i, key.j, key.shift);
        ASSERT_FALSE(abacus.empty()) << "a block the file does not hold was returned";
        DenseBlock want;
        rsatb::io::AbacusBlockToRsatb(orders[layout.atom_type[key.i]], orders[layout.atom_type[key.j]],
                                      abacus.data(), rsatb::io::kAbacusEvPerRydberg, &want);
        ASSERT_EQ(got.rows, want.rows);
        ASSERT_EQ(got.cols, want.cols);
        for (size_t v = 0; v < want.values.size(); ++v) {
            EXPECT_NEAR(got.values[v], want.values[v], 1e-12) << "entry " << v;
        }
    });
    for (size_t k = 0; k < requests.size(); ++k) {
        const BlockKey& key = requests[k];
        const bool exists = key.shift != Shift{{5, 5, 5}} && !f.AbacusBlock(key.i, key.j, key.shift).empty();
        EXPECT_EQ(answered[k], exists ? 1 : 0)
            << "block " << key.i << " " << key.j << " R " << key.shift[0] << key.shift[1] << key.shift[2];
    }
}

TEST(AbacusOrbitalOrder, MatchesTheConventionWrittenOut) {
    // ABACUS p = (z, -x, -y) at positions 0, 1, 2; rsatb p = (y, z, x).
    const AbacusOrbitalOrder p = rsatb::io::MakeAbacusOrbitalOrder({1});
    EXPECT_EQ(p.source, (std::vector<int>{2, 0, 1}));
    EXPECT_EQ(p.sign, (std::vector<double>{-1, 1, -1}));

    // ABACUS d = (z2, -xz, -yz, x2-y2, xy); rsatb d = (xy, yz, z2, xz, x2-y2).
    const AbacusOrbitalOrder d = rsatb::io::MakeAbacusOrbitalOrder({2});
    EXPECT_EQ(d.source, (std::vector<int>{4, 2, 0, 1, 3}));
    EXPECT_EQ(d.sign, (std::vector<double>{1, -1, 1, -1, 1}));

    // Shells are laid out one after another: s p -> s at 0, p at 1..3.
    const AbacusOrbitalOrder sp = rsatb::io::MakeAbacusOrbitalOrder({0, 1});
    EXPECT_EQ(sp.source, (std::vector<int>{0, 3, 1, 2}));
}

class EveryLayout : public ::testing::TestWithParam<std::string> {};

TEST_P(EveryLayout, EachRankGetsExactlyItsShare) {
    const Fixture f;
    OnRootThenBarrier([&] { WriteText(f); WriteBinary(f); WriteNextHam(f, "HR", 1); });
    const std::string& which = GetParam();
    const HamiltonianFile file = which == "nextham"
        ? HamiltonianFile{kNextHam, HamiltonianFormat::kNextHamHdf5}
        : HamiltonianFile{which == "text" ? kText : kBinary, HamiltonianFormat::kAbacusCsr};
    auto reader = rsatb::io::OpenHamiltonianFile(file, FourAtomLayout(), MPI_COMM_WORLD);
    EXPECT_EQ(*reader->metadata().n_orbital_total, 26);
    ExpectFixtureBlocks(*reader, f, ShareOf(f.AllKeys()));

    // A second read, in which only rank 0 asks for anything, still takes
    // every rank through the collective and answers rank 0 in full.
    ExpectFixtureBlocks(*reader, f, Rank() == 0 ? f.AllKeys() : std::vector<BlockKey>{});
}

INSTANTIATE_TEST_SUITE_P(Readers, EveryLayout, ::testing::Values("text", "binary", "nextham"));

TEST(AbacusCsrReader, RejectsABasisOfTheWrongSizeOnEveryRank) {
    const Fixture f;
    OnRootThenBarrier([&] { WriteText(f); WriteBinary(f); });
    auto layout = FourAtomLayout();
    layout.atom_type = {1, 1, 1, 1};  // 36 orbitals, the file has 26
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kText, layout, MPI_COMM_WORLD), std::runtime_error);
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kBinary, layout, MPI_COMM_WORLD), std::runtime_error);
}

TEST(AbacusCsrReader, RejectsComplexTextOnEveryRank) {
    OnRootThenBarrier([] {
        std::ofstream out(kText);
        out << "STEP: 0\nMatrix Dimension of H(R): 26\nMatrix number of H(R): 1\n0 0 0 1\n"
            << " (1.0,0.5)\n 0\n 0";
        for (int a = 0; a < 26; ++a) out << " 1";
        out << "\n";
    });
    rsatb::io::AbacusCsrReader reader(kText, FourAtomLayout(), MPI_COMM_WORLD);
    EXPECT_THROW(reader.ReadBlocks(ShareOf({BlockKey{0, 0, {{0, 0, 0}}}}), [](size_t, const DenseBlock&) {}),
                 std::runtime_error);
}

TEST(AbacusCsrReader, RejectsASpinorLayout) {
    const Fixture f;
    OnRootThenBarrier([&] { WriteText(f); });
    auto layout = FourAtomLayout();
    layout.spin_components = 2;
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kText, layout, MPI_COMM_WORLD), std::runtime_error);
}

TEST(NextHamHdf5Reader, RejectsOverlapSpinorAndMismatchedFiles) {
    const Fixture f;
    auto ignore = [](size_t, const DenseBlock&) {};
    const auto keys = ShareOf(f.AllKeys());

    OnRootThenBarrier([&] { WriteNextHam(f, "SR", 1); });
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, FourAtomLayout()), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);

    OnRootThenBarrier([&] { WriteNextHam(f, "HR", 4); });
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, FourAtomLayout()), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);

    OnRootThenBarrier([&] { WriteNextHam(f, "HR", 1); });
    // Same total size, but per-atom orbital counts that disagree with the file.
    auto swapped = FourAtomLayout();
    swapped.atom_type = {1, 0, 0, 1};
    swapped.type_atomic_number = {14, 6};
    rsatb::io::NextHamHdf5Reader swapped_reader(kNextHam, swapped);
    EXPECT_THROW(swapped_reader.ReadBlocks(keys, ignore), std::runtime_error);

    auto wrong_species = FourAtomLayout();
    wrong_species.type_atomic_number = {6, 32};  // the file records Z = 14
    rsatb::io::NextHamHdf5Reader species_reader(kNextHam, wrong_species);
    EXPECT_THROW(species_reader.ReadBlocks(f.AllKeys(), ignore), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(OpenHamiltonianFile, AbacusOrderFormatsNeedALayout) {
    EXPECT_THROW(rsatb::io::OpenHamiltonianFile(HamiltonianFile{kText, HamiltonianFormat::kAbacusCsr}),
                 std::runtime_error);
    EXPECT_THROW(rsatb::io::OpenHamiltonianFile(HamiltonianFile{kNextHam, HamiltonianFormat::kNextHamHdf5}),
                 std::runtime_error);
    EXPECT_EQ(rsatb::io::ParseHamiltonianFormat(" NextHam "), HamiltonianFormat::kNextHamHdf5);
    EXPECT_EQ(rsatb::io::ParseHamiltonianFormat("abacus"), HamiltonianFormat::kAbacusCsr);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    if (Rank() != 0) {
        delete ::testing::UnitTest::GetInstance()->listeners().Release(
            ::testing::UnitTest::GetInstance()->listeners().default_result_printer());
    }
    const int result = RUN_ALL_TESTS();
    MPI_Barrier(MPI_COMM_WORLD);
    if (Rank() == 0) {
        for (const auto& path : {kText, kBinary, kNextHam}) std::remove(path.c_str());
    }
    int failed = result != 0, any = 0;
    MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any;
}
