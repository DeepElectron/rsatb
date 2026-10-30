#ifndef RSATB_MODEL_HAMILTONIAN_IMAGES_HPP
#define RSATB_MODEL_HAMILTONIAN_IMAGES_HPP

// Reading a precomputed H(R) out of a Hamiltonian file into image blocks.
//
// Two families need this and they are otherwise unalike: LCAO pairs a file
// Hamiltonian with orbital files it computes the overlap from, and tb_file
// takes the same H(R) already orthogonal, with nothing else. What they share
// is exactly this -- "H came from a file" -- so it lives here rather than in
// either of them, and neither has to include the other's machinery.
//
// The reader is an interface (io/hamiltonian_file.hpp), so adding a producer's
// format means adding a reader, not touching this.

#include "../io/hamiltonian_file.hpp"
#include "../operator/images.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace model {

/// Returns the type of every atom in a Hamiltonian file's numbering.
///
/// The file indexes the structure's atoms in their own order, over the
/// primitive cell when the structure is tiled; BuildHamiltonianImagesFromFile
/// folds expanded atom a onto file atom a % primitive_n_atom. A rank holds
/// only its owned and ghost atoms, so the types are combined across ranks.
///
/// Args:
///   ad: The atomic graph.
/// Throws:
///   std::runtime_error if some file atom is held by no rank.
inline std::vector<int> FileAtomTypes(vbcsr::atomic::AtomicData& ad) {
    const auto repetitions = ad.structure_supercell;
    const bool repeated =
        repetitions[0] > 1 || repetitions[1] > 1 || repetitions[2] > 1;
    const int n_file = repeated ? ad.primitive_n_atom : ad.N_atom;

    std::vector<int> types(static_cast<size_t>(n_file), -1);
    for (size_t local = 0; local < ad.atom_type.size(); ++local) {
        const int file_atom = ad.atom_index[local] % n_file;
        types[static_cast<size_t>(file_atom)] = ad.atom_type[local];
    }
    MPI_Allreduce(MPI_IN_PLACE, types.data(), n_file, MPI_INT, MPI_MAX, ad.comm);
    for (int atom = 0; atom < n_file; ++atom) {
        if (types[static_cast<size_t>(atom)] < 0) {
            throw std::runtime_error(
                "Hamiltonian file atom " + std::to_string(atom) + " is held by no rank.");
        }
    }
    return types;
}

/// Builds H(R) by asking the reader for the block behind every graph edge.
///
/// The atomic graph decides which blocks are wanted; the file supplies their
/// values. Each rank collects the distinct file blocks behind its own onsite
/// terms and edges and asks for them in one ReadBlocks call, so the reader can
/// read them in the order its file favours, and a tiled structure reads each
/// primitive block once however many images it lands in. A block whose
/// dimensions disagree with the basis the model declared is an error naming
/// both, because the alternative is a silently truncated Hamiltonian.
///
/// Collective over the reader's communicator: every rank must call it.
///
/// Args:
///   ad: The atomic graph, which fixes the (i, j, R) blocks to ask for.
///   reader: An open Hamiltonian file.
/// Returns:
///   The assembled image container.
/// Throws:
///   std::runtime_error if a block's dimensions disagree with the basis.
template <typename T>
std::unique_ptr<vbcsr::atomic::ImageContainer<T>> BuildHamiltonianImagesFromFileOf(
    vbcsr::atomic::AtomicData& ad,
    const rsatb::io::HamiltonianReader& reader
) {
    auto img = std::make_unique<vbcsr::atomic::ImageContainer<T>>(&ad);

    // Every reader addresses blocks as (i, j, R) over the file's own cell, with
    // atom j in image R, so one mapping folds a tiled structure onto any of
    // them.
    const auto repetitions = ad.structure_supercell;
    const bool repeated =
        repetitions[0] > 1 || repetitions[1] > 1 || repetitions[2] > 1;

    size_t expected_atoms = static_cast<size_t>(ad.primitive_n_atom);
    if (repeated) {
        if (ad.primitive_n_atom <= 0) {
            throw std::runtime_error(
                "File-backed supercell construction requires the primitive atom count.");
        }
        for (int factor : repetitions) {
            if (factor < 1) {
                throw std::runtime_error(
                    "File-backed supercell construction requires positive repetition factors.");
            }
            expected_atoms *= static_cast<size_t>(factor);
        }
        if (expected_atoms != static_cast<size_t>(ad.N_atom)) {
            throw std::runtime_error(
                "File-backed supercell mapping is inconsistent with the expanded atom count.");
        }
    }

    // Expanded atom -> (file atom, image of its primitive cell).
    auto decode_atom = [&](int expanded_atom, int* file_atom, std::array<int, 3>* image) {
        if (!repeated) {
            *file_atom = expanded_atom;
            *image = {{0, 0, 0}};
            return;
        }
        if (expanded_atom < 0 || expanded_atom >= static_cast<int>(expected_atoms)) {
            throw std::runtime_error("Expanded atom index is outside the supercell mapping.");
        }
        *file_atom = expanded_atom % ad.primitive_n_atom;
        int flat_image = expanded_atom / ad.primitive_n_atom;
        (*image)[2] = flat_image % repetitions[2];
        flat_image /= repetitions[2];
        (*image)[1] = flat_image % repetitions[1];
        flat_image /= repetitions[1];
        (*image)[0] = flat_image;
    };

    // Where one file block lands in this rank's images.
    struct Placement {
        rsatb::io::BlockKey key;      // the file block
        std::array<int, 3> shift;     // the graph image it lands in
        int global_i, global_j;
        int inorb, jnorb;
        int atom_i, atom_j;           // expanded indices, for messages
    };
    std::vector<Placement> placements;

    auto place = [&](int local_i, int local_j, const std::array<int, 3>& shift) {
        int inorb, jnorb, atom_i, atom_j;
        ad.get_atom_norb(local_i, &inorb);
        ad.get_atom_norb(local_j, &jnorb);
        if (inorb <= 0 || jnorb <= 0) return;
        ad.get_atom_id(local_i, &atom_i);  // original (unpartitioned) index
        ad.get_atom_id(local_j, &atom_j);

        Placement at;
        std::array<int, 3> image_i, image_j;
        decode_atom(atom_i, &at.key.i, &image_i);
        decode_atom(atom_j, &at.key.j, &image_j);
        for (int axis = 0; axis < 3; ++axis) {
            at.key.shift[axis] = image_j[axis] - image_i[axis] + repetitions[axis] * shift[axis];
        }
        at.shift = shift;
        at.global_i = ad.get_global_index(local_i);
        at.global_j = ad.get_global_index(local_j);
        at.inorb = inorb;
        at.jnorb = jnorb;
        at.atom_i = atom_i;
        at.atom_j = atom_j;
        placements.push_back(at);
    };

    int nstart, nend;
    ad.get_vertex_range_local(&nstart, &nend);
    for (int i = nstart; i < nend; ++i) {
        if (!ad.atom_is_ghost(i)) place(i, i, {{0, 0, 0}});
    }
    int estart, eend;
    ad.get_edge_range_local(&estart, &eend);
    for (int ie = estart; ie < eend; ++ie) {
        int i, j, Rx, Ry, Rz;
        ad.get_connected_atoms(ie, &i, &j);
        ad.get_edge_shift_vec(ie, &Rx, &Ry, &Rz);
        place(i, j, {{Rx, Ry, Rz}});
    }

    // One request per distinct file block; placements first[k] .. first[k + 1]
    // are the ones request k feeds.
    std::sort(placements.begin(), placements.end(),
              [](const Placement& a, const Placement& b) { return a.key < b.key; });
    std::vector<rsatb::io::BlockKey> requests;
    std::vector<size_t> first;
    for (size_t p = 0; p < placements.size(); ++p) {
        if (p == 0 || !(placements[p].key == placements[p - 1].key)) {
            requests.push_back(placements[p].key);
            first.push_back(p);
        }
    }
    first.push_back(placements.size());

    // The first dimension mismatch is recorded and rethrown after the read, so
    // one bad block does not abandon a collective read half-way.
    //
    // Serial within a rank on purpose. HDF5 is built thread-safe here, which
    // is one global mutex around every API call: on diamond sc888 (150528
    // DeePTB datasets) the read took 12.4 s on 1 thread and 55.5 s on 96.
    // Parallelism comes from MPI ranks, which each read only their own blocks.
    std::string mismatch;
    auto place_block = [&](size_t request, const rsatb::io::DenseBlockOf<T>& block) {
        for (size_t p = first[request]; p < first[request + 1]; ++p) {
            const Placement& at = placements[p];
            if (block.rows != at.inorb || block.cols != at.jnorb) {
                if (mismatch.empty()) {
                    mismatch = "Hamiltonian file block (" + std::to_string(at.atom_i) + ", " +
                               std::to_string(at.atom_j) + ") is " + std::to_string(block.rows) +
                               "x" + std::to_string(block.cols) + ", but the model basis expects " +
                               std::to_string(at.inorb) + "x" + std::to_string(at.jnorb) + ".";
                }
                continue;
            }
            img->add_block({at.shift[0], at.shift[1], at.shift[2]}, at.global_i, at.global_j, block.values.data(),
                           at.inorb, at.jnorb, vbcsr::AssemblyMode::ADD,
                           vbcsr::MatrixLayout::RowMajor);
        }
    };
    if constexpr (std::is_same_v<T, double>) {
        reader.ReadBlocks(requests, place_block);
    } else {
        reader.ReadComplexBlocks(requests, place_block);
    }

    if (!mismatch.empty()) {
        throw std::runtime_error(mismatch);
    }

    img->assemble();
    return img;
}

/// Builds H(R) from a file: real, or complex when the reader's blocks are
/// (a spin-orbit H(R) in a spinor basis). See BuildHamiltonianImagesFromFileOf.
inline op::HamiltonianImages BuildHamiltonianImagesFromFile(
    vbcsr::atomic::AtomicData& ad,
    const rsatb::io::HamiltonianReader& reader
) {
    if (reader.complex_blocks()) {
        return BuildHamiltonianImagesFromFileOf<std::complex<double>>(ad, reader);
    }
    return BuildHamiltonianImagesFromFileOf<double>(ad, reader);
}

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_HAMILTONIAN_IMAGES_HPP
