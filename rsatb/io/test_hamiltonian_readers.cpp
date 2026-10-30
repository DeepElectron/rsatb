// Readers for ABACUS CSR and NextHam HDF5 Hamiltonians.
//
// One synthetic H(R), in ABACUS's orbital order and in Rydberg, is written in
// every layout these readers accept -- ABACUS text, ABACUS binary (with the
// older zero-row-offset record for an empty R), and NextHam HDF5 in eV -- and
// each reader must return the same blocks: ABACUS's, put into rsatb's orbital
// order and converted to eV. The orbital map itself is pinned against the
// convention written out by hand.

#include "hamiltonian_readers.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

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
using rsatb::io::DenseBlock;
using rsatb::io::FileBasisLayout;
using rsatb::io::HamiltonianFile;
using rsatb::io::HamiltonianFormat;
using Key = std::array<int, 5>;  // i, j, Rx, Ry, Rz

const std::string kText = "/tmp/rsatb_test_abacus_text.csr";
const std::string kBinary = "/tmp/rsatb_test_abacus_binary.csr";
const std::string kNextHam = "/tmp/rsatb_test_nextham.h5";

// Two species: an s+p atom (4 orbitals) and an s+p+d atom (9 orbitals), Z = 6, 14.
FileBasisLayout TwoAtomLayout() {
    FileBasisLayout layout;
    layout.type_shells = {{0, 1}, {0, 1, 2}};
    layout.type_atomic_number = {6, 14};
    layout.atom_type = {0, 1};
    return layout;
}

// The synthetic H(R): dense random ABACUS-order blocks for a few R, in Ry,
// over the full 13 x 13 basis. R = (0, 0, 1) is present but empty.
struct Fixture {
    int n = 13;
    std::vector<std::array<int, 3>> shifts = {{{0, 0, 0}}, {{1, 0, 0}}, {{-1, 0, 0}},
                                              {{0, 0, 1}}, {{0, -1, 1}}};
    std::map<std::array<int, 3>, std::vector<double>> dense;  // row-major n x n

    Fixture() {
        std::mt19937 engine(7);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        for (const auto& r : shifts) {
            std::vector<double> m(static_cast<size_t>(n * n), 0.0);
            if (r != std::array<int, 3>{{0, 0, 1}}) {
                for (auto& x : m) x = value(engine);
                if (r == std::array<int, 3>{{0, -1, 1}}) {
                    // Only the (0, 1) atom block couples in this image.
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            if (!(a < 4 && b >= 4)) m[a * n + b] = 0.0;
                }
            }
            dense[r] = m;
        }
    }

    int Nnz(const std::array<int, 3>& r) const {
        int count = 0;
        for (double x : dense.at(r)) count += x != 0.0;
        return count;
    }

    // The ABACUS-order block of atoms (i, j), or empty if it is all zero.
    std::vector<double> AbacusBlock(int i, int j, const std::array<int, 3>& r) const {
        const int off[3] = {0, 4, 13};
        std::vector<double> block;
        bool any = false;
        for (int a = off[i]; a < off[i + 1]; ++a)
            for (int b = off[j]; b < off[j + 1]; ++b) {
                block.push_back(dense.at(r)[a * n + b]);
                any = any || block.back() != 0.0;
            }
        return any ? block : std::vector<double>{};
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
        for (int a = 0; a < f.n; ++a) {
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != 0.0) out << " " << m[a * f.n + b];
        }
        out << "\n";
        for (int a = 0; a < f.n; ++a) {
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != 0.0) out << " " << b;
            int row = 0;
            for (int b = 0; b < f.n; ++b) row += m[a * f.n + b] != 0.0;
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
    WriteIntegerAttribute(file, "atom_count", 2);

    std::vector<std::int64_t> pairs, pair_offset{0}, shifts, data_offset{0}, atom_pair_offset{0};
    std::vector<double> data;  // interleaved re, im
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::vector<std::array<int, 3>> present;
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
    WriteInt64(file, "atom_basis_count", {4, 9}, 1);
    WriteInt64(file, "atom_basis_start", {0, 4}, 1);
    WriteInt64(file, "atomic_numbers", {6, 14}, 1);
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

// Every block the fixture holds, through `reader`, against ABACUS's in rsatb order.
void ExpectFixtureBlocks(const rsatb::io::HamiltonianReader& reader, const Fixture& f) {
    const auto layout = TwoAtomLayout();
    const auto orders = rsatb::io::MakeAbacusOrbitalOrders(layout);
    int checked = 0;
    for (const auto& r : f.shifts) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                DenseBlock got;
                const bool found = reader.ReadBlock(i, j, {r[0], r[1], r[2]}, &got);
                const auto abacus = f.AbacusBlock(i, j, r);
                ASSERT_EQ(found, !abacus.empty()) << "block " << i << " " << j << " R "
                                                  << r[0] << r[1] << r[2];
                if (!found) continue;
                DenseBlock want;
                rsatb::io::AbacusBlockToRsatb(orders[layout.atom_type[i]], orders[layout.atom_type[j]],
                                              abacus.data(), rsatb::io::kAbacusEvPerRydberg, &want);
                ASSERT_EQ(got.rows, want.rows);
                ASSERT_EQ(got.cols, want.cols);
                for (size_t k = 0; k < want.values.size(); ++k) {
                    EXPECT_NEAR(got.values[k], want.values[k], 1e-12) << "entry " << k;
                }
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 10);
    DenseBlock unused;
    EXPECT_FALSE(reader.ReadBlock(0, 1, {5, 5, 5}, &unused));
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

TEST(AbacusCsrReader, TextAndBinaryGiveTheSameBlocks) {
    const Fixture f;
    WriteText(f);
    WriteBinary(f);
    const auto layout = TwoAtomLayout();
    for (const auto& path : {kText, kBinary}) {
        SCOPED_TRACE(path);
        auto reader = rsatb::io::OpenHamiltonianFile(HamiltonianFile{path, HamiltonianFormat::kAbacusCsr}, layout);
        EXPECT_EQ(reader->format(), HamiltonianFormat::kAbacusCsr);
        EXPECT_EQ(*reader->metadata().n_orbital_total, 13);
        ExpectFixtureBlocks(*reader, f);
    }
}

TEST(AbacusCsrReader, RejectsABasisOfTheWrongSize) {
    const Fixture f;
    WriteText(f);
    auto layout = TwoAtomLayout();
    layout.atom_type = {1, 1};  // 18 orbitals, the file has 13
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kText, layout), std::runtime_error);
}

TEST(AbacusCsrReader, RejectsComplexText) {
    {
        std::ofstream out(kText);
        out << "STEP: 0\nMatrix Dimension of H(R): 13\nMatrix number of H(R): 1\n0 0 0 1\n"
            << " (1.0,0.5)\n 0\n 0 1 1 1 1 1 1 1 1 1 1 1 1 1\n";
    }
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kText, TwoAtomLayout()), std::runtime_error);
}

TEST(AbacusCsrReader, RejectsASpinorLayout) {
    const Fixture f;
    WriteText(f);
    auto layout = TwoAtomLayout();
    layout.spin_components = 2;
    EXPECT_THROW(rsatb::io::AbacusCsrReader(kText, layout), std::runtime_error);
}

TEST(NextHamHdf5Reader, AgreesWithTheAbacusReader) {
    const Fixture f;
    WriteNextHam(f, "HR", 1);
    auto reader = rsatb::io::OpenHamiltonianFile(HamiltonianFile{kNextHam, HamiltonianFormat::kNextHamHdf5},
                                                 TwoAtomLayout());
    EXPECT_EQ(reader->format(), HamiltonianFormat::kNextHamHdf5);
    ExpectFixtureBlocks(*reader, f);
}

TEST(NextHamHdf5Reader, RejectsOverlapSpinorAndMismatchedFiles) {
    const Fixture f;
    WriteNextHam(f, "SR", 1);
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, TwoAtomLayout()), std::runtime_error);

    WriteNextHam(f, "HR", 4);
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, TwoAtomLayout()), std::runtime_error);

    WriteNextHam(f, "HR", 1);
    auto swapped = TwoAtomLayout();
    swapped.type_shells = {{0, 1, 2}, {0, 1}};  // orbital counts disagree with the file
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, swapped), std::runtime_error);

    auto wrong_species = TwoAtomLayout();
    wrong_species.type_atomic_number = {6, 32};  // the file records Z = 14
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(kNextHam, wrong_species), std::runtime_error);
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
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    for (const auto& path : {kText, kBinary, kNextHam}) std::remove(path.c_str());
    return result;
}
