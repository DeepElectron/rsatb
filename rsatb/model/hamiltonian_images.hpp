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

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <array>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace rsatb {
namespace model {

/// Builds H(R) by asking the reader for the block behind every graph edge.
///
/// The atomic graph decides which blocks are wanted; the file supplies their
/// values. A block whose dimensions disagree with the basis the model declared
/// is an error naming both, because the alternative is a silently truncated
/// Hamiltonian.
///
/// Args:
///   ad: The atomic graph, which fixes the (i, j, R) blocks to ask for.
///   reader: An open Hamiltonian file.
/// Returns:
///   The assembled image container.
/// Throws:
///   std::runtime_error if a block's dimensions disagree with the basis.
inline std::unique_ptr<vbcsr::atomic::ImageContainer<double>> BuildHamiltonianImagesFromFile(
vbcsr::atomic::AtomicData& ad,
const rsatb::io::HamiltonianReader& reader
) {
auto img = std::make_unique<vbcsr::atomic::ImageContainer<double>>(&ad);

    const auto repetitions = ad.structure_supercell;
    const bool repeated =
        repetitions[0] > 1 || repetitions[1] > 1 || repetitions[2] > 1;
    if (repeated && reader.format() != rsatb::io::HamiltonianFormat::kDeeptbHdf5) {
        throw std::runtime_error(
            "Supercell expansion of a file-backed Hamiltonian is currently supported "
            "only for the DeePTB HDF5 (i,j,R) layout; format '" +
            rsatb::io::ToString(reader.format()) + "' has not defined that mapping.");
    }

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

    // A repeated structure asks for the same primitive blocks once per image.
    // Cache those blocks on each rank so geometry expansion does not multiply
    // physical HDF5 reads by the supercell volume.
    using FileKey = std::array<int, 5>;  // i, j, Rx, Ry, Rz
    std::map<FileKey, rsatb::io::DenseBlock> block_cache;
    std::set<FileKey> missing_blocks;

    auto read_block = [&](int atom_i, int atom_j, rsatb::io::CellShift shift,
                          rsatb::io::DenseBlock* block) {
        if (!repeated) {
            return reader.ReadBlock(atom_i, atom_j, shift, block);
        }

        auto decode_atom = [&](int expanded_atom, int* primitive_atom,
                               std::array<int, 3>* image) {
            if (expanded_atom < 0 || expanded_atom >= static_cast<int>(expected_atoms)) {
                throw std::runtime_error(
                    "Expanded atom index is outside the supercell mapping.");
            }
            *primitive_atom = expanded_atom % ad.primitive_n_atom;
            int flat_image = expanded_atom / ad.primitive_n_atom;
            (*image)[2] = flat_image % repetitions[2];
            flat_image /= repetitions[2];
            (*image)[1] = flat_image % repetitions[1];
            flat_image /= repetitions[1];
            (*image)[0] = flat_image;
        };

        int primitive_i = 0;
        int primitive_j = 0;
        std::array<int, 3> image_i{{0, 0, 0}};
        std::array<int, 3> image_j{{0, 0, 0}};
        decode_atom(atom_i, &primitive_i, &image_i);
        decode_atom(atom_j, &primitive_j, &image_j);
        for (int axis = 0; axis < 3; ++axis) {
            shift[axis] = image_j[axis] - image_i[axis] + repetitions[axis] * shift[axis];
        }

        const FileKey key{{primitive_i, primitive_j, shift[0], shift[1], shift[2]}};
        const auto cached = block_cache.find(key);
        if (cached != block_cache.end()) {
            *block = cached->second;
            return true;
        }
        if (missing_blocks.count(key) != 0) {
            return false;
        }
        if (!reader.ReadBlock(primitive_i, primitive_j, shift, block)) {
            missing_blocks.insert(key);
            return false;
        }
        block_cache.emplace(key, *block);
        return true;
    };

    // The first dimension mismatch is recorded and rethrown after both loops,
    // so one bad block does not abort the read half-way. The critical section
    // keeps this correct if these loops are ever threaded again.
    std::string mismatch;
    auto record_mismatch = [&mismatch](const std::string& message) {
        #pragma omp critical(rsatb_hamiltonian_reader_mismatch)
        {
            if (mismatch.empty()) mismatch = message;
        }
    };

    int nstart, nend;
    ad.get_vertex_range_local(&nstart, &nend);

    // Onsite blocks.
    //
    // Deliberately serial. HDF5 is built thread-safe here, which means a single
    // global mutex around every API call: threads gain no concurrency and only
    // contend. Measured on diamond sc888 (150528 datasets), reading this file
    // takes 12.4s on 1 thread and degrades monotonically to 55.5s on 96 --
    // 4.5x slower for doing the same work. The assembly that follows each read
    // is far cheaper than the read, so there is nothing left worth threading.
    //
    // Real parallelism for this file comes from MPI ranks, which own disjoint
    // vertex ranges and open the file independently.
    {
        rsatb::io::DenseBlock block;
        for (int i = nstart; i < nend; ++i) {
            if (ad.atom_is_ghost(i)) continue;

            int inorb;
            ad.get_atom_norb(i, &inorb);
            int atom_idx;
            ad.get_atom_id(i, &atom_idx);  // original (unpartitioned) index

            if (!read_block(atom_idx, atom_idx, {0, 0, 0}, &block)) continue;
            if (block.rows != inorb || block.cols != inorb) {
                record_mismatch(
                    "Hamiltonian file onsite block for atom " + std::to_string(atom_idx) +
                    " is " + std::to_string(block.rows) + "x" + std::to_string(block.cols) +
                    ", but the model basis expects " + std::to_string(inorb) + "x" +
                    std::to_string(inorb) + ".");
                continue;
            }

            const int global_i = ad.get_global_index(i);
            img->add_block({0, 0, 0}, global_i, global_i, block.values.data(),
                           inorb, inorb, vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
        }
    }

    // Edge blocks
    int estart, eend;
    ad.get_edge_range_local(&estart, &eend);
    // Serial for the same reason as the onsite loop above.
    {
        rsatb::io::DenseBlock block;
        for (int ie = estart; ie < eend; ++ie) {
            int i, j;
            ad.get_connected_atoms(ie, &i, &j);

            int inorb, jnorb;
            ad.get_atom_norb(i, &inorb);
            ad.get_atom_norb(j, &jnorb);
            if (inorb <= 0 || jnorb <= 0) continue;

            int Rx, Ry, Rz;
            ad.get_edge_shift_vec(ie, &Rx, &Ry, &Rz);

            int atom_i, atom_j;
            ad.get_atom_id(i, &atom_i);
            ad.get_atom_id(j, &atom_j);

            if (!read_block(atom_i, atom_j, {Rx, Ry, Rz}, &block)) continue;
            if (block.rows != inorb || block.cols != jnorb) {
                record_mismatch(
                    "Hamiltonian file block (" + std::to_string(atom_i) + ", " +
                    std::to_string(atom_j) + ") is " + std::to_string(block.rows) + "x" +
                    std::to_string(block.cols) + ", but the model basis expects " +
                    std::to_string(inorb) + "x" + std::to_string(jnorb) + ".");
                continue;
            }

            const int global_i = ad.get_global_index(i);
            const int global_j = ad.get_global_index(j);
            img->add_block({Rx, Ry, Rz}, global_i, global_j, block.values.data(),
                           inorb, jnorb, vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
        }
    }

    if (!mismatch.empty()) {
        throw std::runtime_error(mismatch);
    }

    img->assemble();
    return img;
}

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_HAMILTONIAN_IMAGES_HPP
