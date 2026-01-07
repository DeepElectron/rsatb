//==========================================================
// AUTHOR : mohan
// DATE : 2008-11-18
//==========================================================
#include <cassert>
#include "memory.h"
#include <iostream>

namespace ModuleBase
{
//    8 bit  = 1 Byte
// 1024 Byte = 1 KB
// 1024 KB   = 1 MB
// 1024 MB   = 1 GB
double Memory::total = 0.0;
int Memory::complex_matrix_memory = 2*sizeof(double); // 16 byte
int Memory::double_memory = sizeof(double); // 8 byte
int Memory::int_memory = sizeof(int); // 4.0 Byte
int Memory::bool_memory = sizeof(bool); // 1.0 Byte
int Memory::float_memory = sizeof(float); // 4.0 Byte
int Memory::short_memory = sizeof(short); // 2.0 Byte

int Memory::n_memory = 1000;
int Memory::n_now = 0;
bool Memory::init_flag = false;

#if defined(__CUDA) || defined(__ROCM)

double Memory::total_gpu = 0.0;
int Memory::n_now_gpu = 0;
bool Memory::init_flag_gpu = false;

std::string *Memory::name_gpu;
std::string *Memory::class_name_gpu;
double *Memory::consume_gpu;

#endif


std::string *Memory::name;
std::string *Memory::class_name;
double *Memory::consume;

Memory::Memory()
{
}

Memory::~Memory()
{
}

double Memory::calculate_mem(const long &n_in,const std::string &type)
{
	double n = static_cast<double>(n_in);
	double mem = 0.0;
	
	double factor = 1.0/1024.0/1024.0;
	double complex_matrix_mem = complex_matrix_memory * factor;
	double double_mem = double_memory * factor;
	double int_mem = int_memory * factor;
	double bool_mem = bool_memory * factor;
	double float_mem = float_memory * factor;
	double short_mem = short_memory * factor;

	if(type=="ModuleBase::ComplexMatrix" || type=="complexmatrix" || type=="cdouble")
	{
		mem = complex_matrix_mem;
	}
	else if(type=="real" || type=="double")
	{
		mem = double_mem;
	}
	else if(type=="int")
	{
		mem = int_mem;
	}
	else if(type=="bool")
	{
		mem = bool_mem;
	}
	else if(type=="short")
	{
		mem = short_mem;
	}
	else if(type=="float")
	{
		mem = float_mem;
	}
	else if(type=="AtomLink")
	{
		mem =  int_mem * 2 + double_mem * 3;
	}
	else if(type=="ModuleBase::Vector3<double>")
	{
		mem = 3 * double_mem;
	}
	else
	{
		std::cout<<"not this type in memory storage : "<<type << std::endl;
	}
	total += n * mem;	
	return n*mem;
}
	

double Memory::record
(
 	const std::string &class_name_in,
	const std::string &name_in,
	const long &n_in,
	const std::string &type,
	const bool accumulate
)
{
	if(!Memory::init_flag)
	{
		name = new std::string[n_memory];
		class_name = new std::string[n_memory];
		consume = new double[n_memory];
		for(int i=0;i<n_memory;i++)
		{
			consume[i] = 0.0;
		}
		Memory::init_flag = true;
	}

	int find = 0;
	for(find = 0; find < n_now; find++)
	{
		if( name_in == name[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now)
	{
		n_now++;
		name[find] = name_in;
		class_name[find] = class_name_in;
	}
	if(n_now >= n_memory)
	{
		std::cout<<" Error! Too many memories required.";
		return 0.0;
	}

	consume[find] = Memory::calculate_mem(n_in,type);

	if(consume[find] > 5)
	{
		std::cout << "\n Warning_Memory_Consuming allocated: " << " "<<name[find]<<" "<<consume[find]<<" MB" << std::endl;
	}
	return consume[find];
}

void Memory::record
(
	const std::string &name_in,
	const size_t &n_in,
	const bool accumulate
)
{
	if(!Memory::init_flag)
	{
		name = new std::string[n_memory];
		class_name = new std::string[n_memory];
		consume = new double[n_memory];
		for(int i=0;i<n_memory;i++)
		{
			consume[i] = 0.0;
		}
		Memory::init_flag = true;
	}

	int find = 0;
	for(find = 0; find < n_now; find++)
	{
		if( name_in == name[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now)
	{
		n_now++;
		name[find] = name_in;
		class_name[find] = "";
	}
	if(n_now >= n_memory)
	{
		std::cout<<" Error! Too many memories has been recorded.";
		return;
	}

	const double factor = 1.0/1024.0/1024.0;
	double size_mb = n_in * factor;

	if(accumulate)
	{
		consume[find] += size_mb;
		Memory::total += size_mb;
	}
	else
	{
		if(consume[find] < size_mb)
		{
			Memory::total += size_mb - consume[find];
			consume[find] = size_mb;
			if(consume[find] > 5)
			{
				std::cout << "\n Warning_Memory_Consuming allocated: " << " "<<name[find]<<" "<<consume[find]<<" MB" << std::endl;
			}
		}
	}

	return;
}

}
