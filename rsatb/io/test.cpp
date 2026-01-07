#include "hdf5.hpp"
#include <gtest/gtest.h>
#include <mpi.h>
#include <iostream>

using namespace rsatb::io;

TEST(ioTest, TestH5ClassInit) {
    std::string path = "../../test/data/H_Al.h5";
    rsatb::io::HDF5Handler hd(path);
}

TEST(ioTest, TestH5GetBlock) {
    std::string path = "../../test/data/H_Al333.h5";
    rsatb::io::HDF5Handler hd(path);
    int i=2,j=0,Rx=0,Ry=0,Rz=1;
    int m,n;
    double *v=nullptr;
    bool found;

    hd.read_from_ijR_dense(i,j,Rx,Ry,Rz, &m,&n,&v,&found);
    if(found){
        for(int i=0;i<m*n;i++) {
            std::cout << v[i] << " ";
            if((i+1)%m == 0){
                std::cout << "\n";
            }
        }
    }

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}