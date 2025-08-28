#ifndef hdf5_io
#define hdf5_io
/* learning materials: https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html */
/* Everything become so hard in c++. */

#include <H5Cpp.h>
#include <iostream>

namespace rsatb::io{

class HDF5Handler
{
    private:
        H5::H5File file;
        std::string FILE_NAME;
        H5::Group group;
    public:
        explicit HDF5Handler(const std::string &FILE_NAME): FILE_NAME(FILE_NAME), file(FILE_NAME, H5F_ACC_RDONLY), group(file.openGroup("0"))
        {}

        int read_from_ijR_dense(
            int i, int j, int Rx, int Ry, int Rz,
            // output
            int *m, // rowdim
            int *n, // col dim
            double **v, // value
            bool *found
        ) // handling the ijR format of the h5 file.
        {   
            bool Transpose = false;
            *found = true;
            std::string result = std::to_string(i) + "_" + 
                         std::to_string(j) + "_" + 
                         std::to_string(Rx) + "_" + 
                         std::to_string(Ry) + "_" + 
                         std::to_string(Rz);

            H5::DataSet dataset;
            H5std_string DATASET_NAME(result);
            bool exist = group.exists(DATASET_NAME);
            try
            {
                if(exist){
                    dataset = group.openDataSet(DATASET_NAME);
                }else{
                    result = std::to_string(j) + "_" + 
                            std::to_string(i) + "_" + 
                            std::to_string(-Rx) + "_" + 
                            std::to_string(-Ry) + "_" + 
                            std::to_string(-Rz);
                            
                    H5std_string DATASET_NAME_T(result);
                    bool exist_T = group.exists(DATASET_NAME_T);
                    if(!exist_T) {
                        *found = false;
                        return 0;
                    }
                    dataset = group.openDataSet(DATASET_NAME_T);
                }

                H5T_class_t type_class = dataset.getTypeClass();

                H5::DataSpace dataspace = dataset.getSpace();
                int rank = dataspace.getSimpleExtentNdims();
                if (rank != 2) {
                    throw std::runtime_error("The data structure is not supported !");
                }
                hsize_t dims[2];
                dataspace.getSimpleExtentDims(dims, NULL);
                /* rank is the number of dim, dims is the size of each axis */
                if(exist){
                    *m = static_cast<int>(dims[0]);
                    *n = static_cast<int>(dims[1]);
                } else {
                    *m = static_cast<int>(dims[1]);
                    *n = static_cast<int>(dims[0]);
                }
                
                // float data_out[*m][*n];

                // for (j = 0; j < *m; j++)
                // {
                //     for (i = 0; i < *n; i++)
                //     {
                //         data_out[j][i] = 0;
                //     }
                // }
                *v = new double[(*m)*(*n)];
                if(exist){
                    H5::DataSpace memspace(2, dims);
                    dataset.read(*v, H5::PredType::NATIVE_DOUBLE, memspace, dataspace);

                }else{
                    hsize_t start[2]  = {0,0};
                    hsize_t count[2]  = {dims[0],1};
                    H5::DataSpace memspace(2, count);

                    for(int ncol=0; ncol<dims[1]; ncol++){
                        start[1] = ncol;
                        dataspace.selectHyperslab(H5S_SELECT_SET, count, start, NULL, NULL);
                        // mem_space.selectHyperslab(H5S_SELECT_SET, );
                        dataset.read(*v+ncol*dims[0], H5::PredType::NATIVE_DOUBLE, memspace, dataspace);
                    }
                }
                // *v = data_out
            }
            catch(H5::FileIException error)
            {
                error.printErrorStack();
                return -1;
            }
            // catch failure caused by the DataSet operations
            catch(H5::DataSetIException error)
            {
                error.printErrorStack();
                return -1;
            }
            // catch failure caused by the DataSpace operations
            catch(H5::DataSpaceIException error)
            {
                error.printErrorStack();
                return -1;
            }
            // catch failure caused by the DataSpace operations
            catch(H5::DataTypeIException error)
            {
                error.printErrorStack();
                return -1;
            }

            return 0;
        };

        // void read_from_R_csr(
        //     int Rx, int Ry, int Rz,
        //     // output
        //     int *n, // nxn mat size
        //     int *indices,
        //     int *intptr,
        //     double *data
        //     );      
};
}

#endif