
import unittest
import numpy as np
import sys
import os

# Ensure rsatb is in path
sys.path.append(os.getcwd())

try:
    from rsatb.utils.parrallel import MPIMultiLevel, MPIEnv
except ImportError:
    # Fallback if MPIEnv alias is not yet available
    from rsatb.utils.parrallel import MPIMultiLevel
    MPIEnv = MPIMultiLevel

class TestMPIEnv(unittest.TestCase):
    def setUp(self):
        # Determine if we are in an MPI environment
        self.mpi_available = False
        try:
            from mpi4py import MPI
            self.mpi_available = True
        except ImportError:
            pass
        
        # Initialize with MPI enabled if available, else disabled
        self.env = MPIEnv(enable_mpi=self.mpi_available, verbose=False) # verbose=False to reduce noise

    def test_initialization(self):
        if self.env.rank == 0:
            print(f"\n[Test Initialization] Rank: {self.env.rank}, Size: {self.env.size}")
        self.assertIsInstance(self.env.rank, int)
        self.assertIsInstance(self.env.size, int)
        if not self.mpi_available:
            self.assertEqual(self.env.size, 1)
            self.assertEqual(self.env.rank, 0)

    def test_partitioning(self):
        if self.env.rank == 0:
            print("\n[Test Partitioning]")
        # Test default partition (n_group=1)
        self.env.partition(n_group=1)
        self.assertEqual(self.env._MPIMultiLevel__n_group, 1)
        
        # Test invalid partition
        with self.assertRaises(ValueError):
            self.env.partition(n_group=self.env.size + 1)

    def test_dist_list(self):
        if self.env.rank == 0:
            print("\n[Test dist_list]")
        raw_list = list(range(10))
        
        # Level 2 (Global)
        sublist = self.env.dist_list(raw_list, level=2)
        self.assertIsInstance(sublist, list)
        if self.env.size == 1:
            self.assertEqual(sublist, raw_list)
            
        # Level 0 (Submaster)
        sublist_l0 = self.env.dist_list(raw_list, level=0)
        if self.env.size == 1:
             self.assertEqual(sublist_l0, raw_list)

    def test_dist_range(self):
        if self.env.rank == 0:
            print("\n[Test dist_range]")
        n_max = 10
        
        # Level 2
        r = self.env.dist_range(n_max, level=2)
        self.assertIsInstance(r, range)
        if self.env.size == 1:
            self.assertEqual(list(r), list(range(n_max)))
            
        # Level 0
        r0 = self.env.dist_range(n_max, level=0)
        self.assertIsNotNone(r0)
        if self.env.size == 1:
            self.assertEqual(list(r0), list(range(n_max)))

    def test_collectives_reduce(self):
        if self.env.rank == 0:
            print("\n[Test Reduce]")
        data = np.array([1.0, 2.0])
        
        # Level 2
        res = self.env.reduce(data, level=2)
        if self.env.is_master:
            self.assertEqual(res.shape, data.shape)
            # Expected sum is data * size
            expected = data * self.env.size
            np.testing.assert_allclose(res, expected)
        
        # Level 0
        res0 = self.env.reduce(data, level=0)
        if self.env.size == 1:
            np.testing.assert_array_equal(res0, data)

    def test_collectives_gather(self):
        if self.env.rank == 0:
            print("\n[Test Gather]")
        data = np.array([1.0, 2.0])
        
        # Level 2
        res = self.env.gather(data, level=2)
        if self.env.is_master:
            expected_len = data.shape[0] * self.env.size
            self.assertEqual(res.shape[0], expected_len)
            
        # Level 0
        res0 = self.env.gather(data, level=0)
        if self.env.size == 1:
            np.testing.assert_array_equal(res0, data)

    def test_collectives_bcast(self):
        if self.env.rank == 0:
            print("\n[Test Bcast]")
        data = np.array([1.0, 2.0])
        self.env.bcast(data, level=2)
        self.env.bcast(data, level=0)

    def test_multigroup_collectives(self):
        if self.env.size < 2:
            if self.env.rank == 0:
                print("\n[Skipping Multi-group Test] Need at least 2 ranks")
            return

        if self.env.rank == 0:
            print(f"\n[Test Multi-group Collectives] Size: {self.env.size}, Groups: 2")
        
        # Partition into 2 groups
        self.env.partition(n_group=2)
        
        # --- Level 1: Subgroup Reduce ---
        # Each rank has data = [1.0]
        data = np.array([1.0])
        res_l1 = self.env.reduce(data, level=1)
        
        # Check result on submaster
        if self.env.is_submaster:
            # Expected: sum of 1.0 for each member in subgroup
            expected = data * self.env.subsize
            np.testing.assert_allclose(res_l1, expected, err_msg=f"Level 1 Reduce Failed on Rank {self.env.rank}")
            # print(f"  Rank {self.env.rank} (Submaster): Level 1 Reduce OK. Got {res_l1}")

        # --- Level 0: Submaster Reduce ---
        # Submasters have data = [1.0] (reset data)
        data = np.array([1.0])
        res_l0 = self.env.reduce(data, level=0)
        
        # Check result on global master
        if self.env.is_master:
            # Expected: sum of 1.0 for each submaster (which is n_group=2)
            expected = data * 2 # n_group=2
            np.testing.assert_allclose(res_l0, expected, err_msg="Level 0 Reduce Failed")
            # print(f"  Rank {self.env.rank} (Master): Level 0 Reduce OK. Got {res_l0}")

        # --- Level 2: Global Reduce ---
        data = np.array([1.0])
        res_l2 = self.env.reduce(data, level=2)
        if self.env.is_master:
            expected = data * self.env.size
            np.testing.assert_allclose(res_l2, expected, err_msg="Level 2 Reduce Failed")
            # print(f"  Rank {self.env.rank} (Master): Level 2 Reduce OK. Got {res_l2}")

        # --- Level 1: Subgroup Bcast ---
        if self.env.is_submaster:
            data_bcast = np.array([float(self.env.color + 10)]) # Group 0 -> 10.0, Group 1 -> 11.0
        else:
            data_bcast = np.array([0.0])
        
        self.env.bcast(data_bcast, level=1)
        
        # Verify everyone has their group's value
        expected_val = float(self.env.color + 10)
        np.testing.assert_allclose(data_bcast, [expected_val], err_msg=f"Level 1 Bcast Failed on Rank {self.env.rank}")
        
        # Barrier to sync
        self.env.barrier(level=2)
        if self.env.rank == 0:
            print("  Multi-group tests passed.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
