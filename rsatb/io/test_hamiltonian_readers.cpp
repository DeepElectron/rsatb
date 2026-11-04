// Readers for ABACUS CSR and NextHam HDF5 Hamiltonians.
//
// One synthetic H(R), in ABACUS's orbital order and in Rydberg, is written in
// every layout these readers accept -- ABACUS text, ABACUS binary (with the
// older zero-row-offset record for an empty R), and NextHam HDF5 in eV -- and
// each reader must return the same blocks: ABACUS's, put into rsatb's orbital
// order and converted to eV. That is done twice: spin-degenerate (real
// values) and spinor (nspin=4: complex values, spin interleaved inside each
// orbital, twice the orbitals). The orbital map itself is pinned against the
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
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using rsatb::io::AbacusOrbitalOrder;
using rsatb::io::BlockKey;
using rsatb::io::DenseBlockOf;
using rsatb::io::FileBasisLayout;
using rsatb::io::HamiltonianFile;
using rsatb::io::HamiltonianFormat;
using Complex = std::complex<double>;
using Shift = std::array<int, 3>;

template <typename V>
constexpr bool kSpinor = std::is_same_v<V, Complex>;

template <typename V>
std::string Path(const std::string& what) {
    return "/tmp/rsatb_test_" + what + (kSpinor<V> ? "_spinor" : "");
}
template <typename V> std::string TextPath() { return Path<V>("abacus_text.csr"); }
template <typename V> std::string BinaryPath() { return Path<V>("abacus_binary.csr"); }
template <typename V> std::string NextHamPath() { return Path<V>("nextham.h5"); }

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

// Two species, an s+p atom (4 spatial orbitals, Z = 6) and an s+p+d atom (9,
// Z = 14), as atoms C Si Si C: a 26-orbital basis, 52 as spinors.
template <typename V>
FileBasisLayout FourAtomLayout() {
    FileBasisLayout layout;
    layout.type_shells = {{0, 1}, {0, 1, 2}};
    layout.type_atomic_number = {6, 14};
    layout.atom_type = {0, 1, 1, 0};
    layout.spin_components = kSpinor<V> ? 2 : 1;
    return layout;
}

// The synthetic H(R): random ABACUS-order matrices over the basis, in Ry,
// with a third of the entries zero so rows are genuinely sparse. R = (0, 0, 1)
// is present but empty, and R = (0, -1, 1) couples only atom 0 to atoms 1, 2.
template <typename V>
struct Fixture {
    int c = kSpinor<V> ? 2 : 1;
    int n = 26 * c;
    std::array<int, 5> offsets{{0, 4 * c, 13 * c, 22 * c, 26 * c}};
    std::vector<Shift> shifts = {{{0, 0, 0}}, {{1, 0, 0}}, {{-1, 0, 0}}, {{0, 0, 1}}, {{0, -1, 1}}};
    std::map<Shift, std::vector<V>> dense;  // row-major n x n

    Fixture() {
        std::mt19937 engine(7);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        for (const auto& r : shifts) {
            std::vector<V> m(static_cast<size_t>(n * n), V(0));
            if (r != Shift{{0, 0, 1}}) {
                for (auto& x : m) {
                    const double re = value(engine);
                    if (std::abs(re) < 0.33) continue;
                    if constexpr (kSpinor<V>) x = V(re, value(engine)); else x = re;
                }
                if (r == Shift{{0, -1, 1}}) {
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            if (!(a < offsets[1] && b >= offsets[1] && b < offsets[3])) m[a * n + b] = V(0);
                }
            }
            dense[r] = m;
        }
    }

    int Nnz(const Shift& r) const {
        int count = 0;
        for (const V& x : dense.at(r)) count += x != V(0);
        return count;
    }

    // The ABACUS-order block of atoms (i, j), or empty if it is all zero.
    std::vector<V> AbacusBlock(int i, int j, const Shift& r) const {
        std::vector<V> block;
        bool any = false;
        for (int a = offsets[i]; a < offsets[i + 1]; ++a)
            for (int b = offsets[j]; b < offsets[j + 1]; ++b) {
                block.push_back(dense.at(r)[a * n + b]);
                any = any || block.back() != V(0);
            }
        return any ? block : std::vector<V>{};
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

// ABACUS's text value: "x", or "(re,im)" for nspin=4.
std::string TextValue(double x) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", x);
    return buffer;
}
std::string TextValue(const Complex& x) {
    return "(" + TextValue(x.real()) + "," + TextValue(x.imag()) + ")";
}

template <typename V>
void WriteText(const Fixture<V>& f) {
    std::ofstream out(TextPath<V>());
    out << "STEP: 0\nMatrix Dimension of H(R): " << f.n << "\nMatrix number of H(R): "
        << f.shifts.size() << "\n";
    for (const auto& r : f.shifts) {
        out << r[0] << " " << r[1] << " " << r[2] << " " << f.Nnz(r) << "\n";
        if (f.Nnz(r) == 0) continue;  // current ABACUS writes nothing more
        const auto& m = f.dense.at(r);
        std::vector<int> offsets{0};
        for (int a = 0; a < f.n * f.n; ++a)
            if (m[a] != V(0)) out << " " << TextValue(m[a]);
        out << "\n";
        for (int a = 0; a < f.n; ++a) {
            int row = 0;
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != V(0)) {
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

template <typename V>
void WriteBinary(const Fixture<V>& f) {
    std::ofstream out(BinaryPath<V>(), std::ios::binary);
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
        for (const V& x : m)
            if (x != V(0)) out.write(reinterpret_cast<const char*>(&x), sizeof(V));
        for (int a = 0; a < f.n; ++a)
            for (int b = 0; b < f.n; ++b)
                if (m[a * f.n + b] != V(0)) put(b);
        std::int32_t total = 0;
        put(0);
        for (int a = 0; a < f.n; ++a) {
            for (int b = 0; b < f.n; ++b) total += m[a * f.n + b] != V(0);
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
template <typename V>
void WriteNextHam(const Fixture<V>& f, const std::string& op, long long nspin) {
    H5::H5File file(NextHamPath<V>(), H5F_ACC_TRUNC);
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
                for (const V& x : f.AbacusBlock(i, j, r)) {
                    const Complex z = Complex(x) * rsatb::io::kAbacusEvPerRydberg;
                    data.push_back(z.real());
                    data.push_back(z.imag());
                }
            }
            pair_offset.push_back(static_cast<std::int64_t>(shifts.size() / 3));
            data_offset.push_back(static_cast<std::int64_t>(data.size() / 2));
        }
        atom_pair_offset.push_back(static_cast<std::int64_t>(pairs.size() / 2));
    }
    const std::int64_t c = f.c;
    WriteInt64(file, "atom_basis_count", {4 * c, 9 * c, 9 * c, 4 * c}, 1);
    WriteInt64(file, "atom_basis_start", {0, 4 * c, 13 * c, 22 * c}, 1);
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

template <typename V>
void Read(const rsatb::io::HamiltonianReader& reader, const std::vector<BlockKey>& requests,
          const rsatb::io::BlockSinkOf<V>& sink) {
    if constexpr (kSpinor<V>) reader.ReadComplexBlocks(requests, sink);
    else reader.ReadBlocks(requests, sink);
}

// Reads `requests`, and checks every answer against ABACUS's block in rsatb order.
template <typename V>
void ExpectFixtureBlocks(const rsatb::io::HamiltonianReader& reader, const Fixture<V>& f,
                         const std::vector<BlockKey>& requests) {
    const auto layout = FourAtomLayout<V>();
    const auto orders = rsatb::io::MakeAbacusOrbitalOrders(layout);
    EXPECT_EQ(reader.complex_blocks(), kSpinor<V>);
    std::vector<int> answered(requests.size(), 0);
    Read<V>(reader, requests, [&](size_t k, const DenseBlockOf<V>& got) {
        ++answered[k];
        const BlockKey& key = requests[k];
        const auto abacus = f.AbacusBlock(key.i, key.j, key.shift);
        ASSERT_FALSE(abacus.empty()) << "a block the file does not hold was returned";
        DenseBlockOf<V> want;
        rsatb::io::AbacusBlockToRsatb(orders[layout.atom_type[key.i]], orders[layout.atom_type[key.j]],
                                      abacus.data(), rsatb::io::kAbacusEvPerRydberg, &want);
        ASSERT_EQ(got.rows, want.rows);
        ASSERT_EQ(got.cols, want.cols);
        for (size_t v = 0; v < want.values.size(); ++v) {
            EXPECT_NEAR(std::abs(got.values[v] - want.values[v]), 0.0, 1e-12) << "entry " << v;
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

    // Spinor: spin rides inside each orbital, 2 * orbital + s on both sides.
    const AbacusOrbitalOrder p2 = rsatb::io::MakeAbacusOrbitalOrder({1}, 2);
    EXPECT_EQ(p2.source, (std::vector<int>{4, 5, 0, 1, 2, 3}));
    EXPECT_EQ(p2.sign, (std::vector<double>{-1, -1, 1, 1, -1, -1}));
}

class EveryLayout : public ::testing::TestWithParam<std::string> {};

template <typename V>
void EachRankGetsExactlyItsShare(const std::string& which) {
    const Fixture<V> f;
    OnRootThenBarrier([&] { WriteText(f); WriteBinary(f); WriteNextHam(f, "HR", kSpinor<V> ? 4 : 1); });
    const HamiltonianFile file = which == "nextham"
        ? HamiltonianFile{NextHamPath<V>(), HamiltonianFormat::kNextHamHdf5}
        : HamiltonianFile{which == "text" ? TextPath<V>() : BinaryPath<V>(), HamiltonianFormat::kAbacusCsr};
    auto reader = rsatb::io::OpenHamiltonianFile(file, FourAtomLayout<V>(), MPI_COMM_WORLD);
    EXPECT_EQ(*reader->metadata().n_orbital_total, f.n);
    ExpectFixtureBlocks(*reader, f, ShareOf(f.AllKeys()));

    // A second read, in which only rank 0 asks for anything, still takes
    // every rank through the collective and answers rank 0 in full.
    ExpectFixtureBlocks(*reader, f, Rank() == 0 ? f.AllKeys() : std::vector<BlockKey>{});
}

TEST_P(EveryLayout, EachRankGetsExactlyItsShare) {
    SCOPED_TRACE("spin-degenerate");
    EachRankGetsExactlyItsShare<double>(GetParam());
}

TEST_P(EveryLayout, EachRankGetsExactlyItsShareAsSpinor) {
    SCOPED_TRACE("spinor");
    EachRankGetsExactlyItsShare<Complex>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Readers, EveryLayout, ::testing::Values("text", "binary", "nextham"));

TEST(AbacusCsrReader, RejectsABasisOfTheWrongSizeOnEveryRank) {
    const Fixture<double> f;
    OnRootThenBarrier([&] { WriteText(f); WriteBinary(f); });
    auto layout = FourAtomLayout<double>();
    layout.atom_type = {1, 1, 1, 1};  // 36 orbitals, the file has 26
    EXPECT_THROW(rsatb::io::AbacusCsrReader(TextPath<double>(), layout, MPI_COMM_WORLD), std::runtime_error);
    EXPECT_THROW(rsatb::io::AbacusCsrReader(BinaryPath<double>(), layout, MPI_COMM_WORLD), std::runtime_error);
}

TEST(AbacusCsrReader, RejectsValuesOfTheWrongKindOnEveryRank) {
    const Fixture<double> real;
    const Fixture<Complex> spinor;
    OnRootThenBarrier([&] { WriteText(real); WriteText(spinor); WriteBinary(spinor); });
    auto ignore_real = [](size_t, const rsatb::io::DenseBlock&) {};
    auto ignore_complex = [](size_t, const rsatb::io::ComplexDenseBlock&) {};

    // Complex nspin=4 text read as spin-degenerate: the 52-orbital file does
    // not fit the 26-orbital basis.
    EXPECT_THROW(rsatb::io::AbacusCsrReader(TextPath<Complex>(), FourAtomLayout<double>(), MPI_COMM_WORLD),
                 std::runtime_error);
    // Real text read as spinor, under a spinor layout of the same total size
    // (13 spatial orbitals, 26 as spinors), so the dimension passes and the
    // value parser has to notice the missing "(re,im)".
    FileBasisLayout fitting;
    fitting.type_shells = {{0, 1}, {0, 1, 2}};
    fitting.type_atomic_number = {6, 14};
    fitting.atom_type = {1, 0};  // 9 + 4 = 13 spatial, 26 as spinors: the real file's size
    fitting.spin_components = 2;
    rsatb::io::AbacusCsrReader real_as_spinor(TextPath<double>(), fitting, MPI_COMM_WORLD);
    EXPECT_THROW(real_as_spinor.ReadComplexBlocks(ShareOf({BlockKey{0, 0, {{0, 0, 0}}}}), ignore_complex),
                 std::runtime_error);

    // A spinor reader is not asked for real blocks, nor a real one for complex.
    rsatb::io::AbacusCsrReader spinor_reader(BinaryPath<Complex>(), FourAtomLayout<Complex>(), MPI_COMM_WORLD);
    EXPECT_THROW(spinor_reader.ReadBlocks({}, ignore_real), std::runtime_error);
}

TEST(NextHamHdf5Reader, RejectsOverlapSpinMismatchAndMismatchedFiles) {
    const Fixture<double> f;
    auto ignore = [](size_t, const rsatb::io::DenseBlock&) {};
    const auto keys = ShareOf(f.AllKeys());

    OnRootThenBarrier([&] { WriteNextHam(f, "SR", 1); });
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(NextHamPath<double>(), FourAtomLayout<double>()), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);

    // An nspin=4 file read as spin-degenerate, and an nspin=1 one as spinor.
    OnRootThenBarrier([&] { WriteNextHam(f, "HR", 4); });
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(NextHamPath<double>(), FourAtomLayout<double>()), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);
    OnRootThenBarrier([&] { WriteNextHam(f, "HR", 1); });
    EXPECT_THROW(rsatb::io::NextHamHdf5Reader(NextHamPath<double>(), FourAtomLayout<Complex>()), std::runtime_error);

    // Same total size, but per-atom orbital counts that disagree with the file.
    auto swapped = FourAtomLayout<double>();
    swapped.atom_type = {1, 0, 0, 1};
    swapped.type_atomic_number = {14, 6};
    rsatb::io::NextHamHdf5Reader swapped_reader(NextHamPath<double>(), swapped);
    EXPECT_THROW(swapped_reader.ReadBlocks(keys, ignore), std::runtime_error);

    auto wrong_species = FourAtomLayout<double>();
    wrong_species.type_atomic_number = {6, 32};  // the file records Z = 14
    rsatb::io::NextHamHdf5Reader species_reader(NextHamPath<double>(), wrong_species);
    EXPECT_THROW(species_reader.ReadBlocks(f.AllKeys(), ignore), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);
}

// DeePTB, as its ABACUS parser writes it: an atom of 2 spatial orbitals and one
// of 1, H in single-precision compound complex, spin-blocked, one direction of
// the Hermitian pair stored. Entry value (si, oi, sj, oj) encodes its own indices.
const std::string kDeeptbSpinorPath = "/tmp/rsatb_test_deeptb_spinor.h5";

Complex SpinorEntry(int si, int oi, int sj, int oj) {
    return Complex(1000 * si + 100 * oi + 10 * sj + oj, 1 + si + 2 * oi + 4 * sj + 8 * oj);
}

void WriteDeeptbSpinor() {
    if (Rank() != 0) return;
    struct C64 { float r, i; };
    H5::CompType type(sizeof(C64));
    type.insertMember("r", 0, H5::PredType::NATIVE_FLOAT);
    type.insertMember("i", sizeof(float), H5::PredType::NATIVE_FLOAT);
    H5::H5File file(kDeeptbSpinorPath, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("0");
    const int n_i = 2, n_j = 1;
    std::vector<C64> block(static_cast<size_t>(2 * n_i) * 2 * n_j);
    for (int si = 0; si < 2; ++si)
        for (int oi = 0; oi < n_i; ++oi)
            for (int sj = 0; sj < 2; ++sj)
                for (int oj = 0; oj < n_j; ++oj) {
                    const Complex v = SpinorEntry(si, oi, sj, oj);
                    block[static_cast<size_t>(si * n_i + oi) * (2 * n_j) + sj * n_j + oj] =
                        C64{static_cast<float>(v.real()), static_cast<float>(v.imag())};
                }
    const hsize_t dims[2] = {static_cast<hsize_t>(2 * n_i), static_cast<hsize_t>(2 * n_j)};
    group.createDataSet("0_1_1_0_-2", type, H5::DataSpace(2, dims)).write(block.data(), type);
    // A spin-degenerate-sized block, for the rejection below.
    const hsize_t odd[2] = {3, 3};
    std::vector<C64> three(9, C64{1.0f, 0.0f});
    group.createDataSet("2_2_0_0_0", type, H5::DataSpace(2, odd)).write(three.data(), type);
    // Real, in single precision, as overlaps.h5 holds it.
    const hsize_t real_dims[2] = {2, 1};
    const float real_values[2] = {0.25f, -0.5f};
    group.createDataSet("3_4_0_1_0", H5::PredType::NATIVE_FLOAT, H5::DataSpace(2, real_dims))
        .write(real_values, H5::PredType::NATIVE_FLOAT);
}

TEST(DeeptbHdf5Reader, SpinorBlocksArriveInterleavedAndConjugateTransposed) {
    WriteDeeptbSpinor();
    MPI_Barrier(MPI_COMM_WORLD);
    FileBasisLayout layout;
    layout.spin_components = 2;
    auto reader = rsatb::io::OpenHamiltonianFile(
        HamiltonianFile{kDeeptbSpinorPath, HamiltonianFormat::kDeeptbHdf5}, layout, MPI_COMM_WORLD);
    ASSERT_TRUE(reader->complex_blocks());

    BlockKey stored;
    stored.i = 0; stored.j = 1; stored.shift = {{1, 0, -2}};
    BlockKey partner;
    partner.i = 1; partner.j = 0; partner.shift = {{-1, 0, 2}};
    BlockKey absent;
    absent.i = 0; absent.j = 1; absent.shift = {{5, 5, 5}};

    int seen = 0;
    reader->ReadComplexBlocks({stored, partner, absent}, [&](size_t request, const DenseBlockOf<Complex>& block) {
        ++seen;
        ASSERT_NE(request, 2u);
        const int n_i = 2, n_j = 1;
        for (int si = 0; si < 2; ++si)
            for (int oi = 0; oi < n_i; ++oi)
                for (int sj = 0; sj < 2; ++sj)
                    for (int oj = 0; oj < n_j; ++oj) {
                        const Complex want = SpinorEntry(si, oi, sj, oj);
                        const int row = 2 * oi + si, col = 2 * oj + sj;
                        if (request == 0) {
                            ASSERT_EQ(block.rows, 4);
                            ASSERT_EQ(block.cols, 2);
                            EXPECT_EQ(block.values[static_cast<size_t>(row) * block.cols + col], want);
                        } else {
                            ASSERT_EQ(block.rows, 2);
                            ASSERT_EQ(block.cols, 4);
                            EXPECT_EQ(block.values[static_cast<size_t>(col) * block.cols + row], std::conj(want));
                        }
                    }
    });
    EXPECT_EQ(seen, 2);
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(DeeptbHdf5Reader, RejectsABlockNoSpinorBasisCanHold) {
    FileBasisLayout layout;
    layout.spin_components = 2;
    auto reader = rsatb::io::OpenHamiltonianFile(
        HamiltonianFile{kDeeptbSpinorPath, HamiltonianFormat::kDeeptbHdf5}, layout, MPI_COMM_WORLD);
    BlockKey odd;
    odd.i = 2; odd.j = 2;
    EXPECT_THROW(reader->ReadComplexBlocks({odd}, [](size_t, const DenseBlockOf<Complex>&) {}),
                 std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(DeeptbHdf5Reader, ReadsRealSinglePrecisionBlocksAndKeepsSpinApart) {
    auto reader = rsatb::io::OpenHamiltonianFile(
        HamiltonianFile{kDeeptbSpinorPath, HamiltonianFormat::kDeeptbHdf5});
    EXPECT_FALSE(reader->complex_blocks());
    BlockKey key;
    key.i = 3; key.j = 4; key.shift = {{0, 1, 0}};
    BlockKey partner;
    partner.i = 4; partner.j = 3; partner.shift = {{0, -1, 0}};
    int seen = 0;
    reader->ReadBlocks({key, partner}, [&](size_t request, const DenseBlockOf<double>& block) {
        ++seen;
        EXPECT_EQ(block.values.size(), 2u);
        EXPECT_EQ(block.values[0], 0.25);
        EXPECT_EQ(block.values[1], -0.5);
        EXPECT_EQ(block.rows, request == 0 ? 2 : 1);
    });
    EXPECT_EQ(seen, 2);
    // A real-basis reader does not offer complex blocks, nor a spinor one real ones.
    EXPECT_THROW(reader->ReadComplexBlocks({key}, [](size_t, const DenseBlockOf<Complex>&) {}),
                 std::runtime_error);
    auto spinor = rsatb::io::OpenHamiltonianFile(
        HamiltonianFile{kDeeptbSpinorPath, HamiltonianFormat::kDeeptbHdf5}, 2);
    EXPECT_THROW(spinor->ReadBlocks({key}, [](size_t, const DenseBlockOf<double>&) {}), std::runtime_error);
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(OpenHamiltonianFile, AbacusOrderFormatsNeedALayout) {
    EXPECT_THROW(rsatb::io::OpenHamiltonianFile(HamiltonianFile{TextPath<double>(), HamiltonianFormat::kAbacusCsr}),
                 std::runtime_error);
    EXPECT_THROW(rsatb::io::OpenHamiltonianFile(HamiltonianFile{NextHamPath<double>(), HamiltonianFormat::kNextHamHdf5}),
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
        for (const auto& path : {TextPath<double>(), BinaryPath<double>(), NextHamPath<double>(),
                                 TextPath<Complex>(), BinaryPath<Complex>(), NextHamPath<Complex>(),
                                 kDeeptbSpinorPath}) {
            std::remove(path.c_str());
        }
    }
    int failed = result != 0, any = 0;
    MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any;
}
