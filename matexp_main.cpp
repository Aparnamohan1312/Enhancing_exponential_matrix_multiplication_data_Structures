#include <cstdlib>
#include "archlab.h"
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include "function_map.hpp"
#include <dlfcn.h>
#include "tensor_t.hpp"
#include "perfstats.h"

#define ELEMENT_TYPE uint64_t

uint array_size;

typedef void(*matexp_impl)(tensor_t<ELEMENT_TYPE> & , const tensor_t<ELEMENT_TYPE> &, uint32_t power, uint64_t seed, int iterations);

int main(int argc, char *argv[])
{

	
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	std::vector<int> powers;
	std::vector<int> default_powers;
	int i, reps=1, size, iterations=1,mhz, power, verify =0;
    char *stat_file = NULL;
    char default_filename[] = "stats.csv";
    char preamble[1024];
    char epilogue[1024];
    char header[1024];
	std::stringstream clocks;
	std::vector<std::string> functions;
	std::vector<std::string> default_functions;
	std::vector<unsigned long int> sizes;
	std::vector<unsigned long int> default_sizes;
	default_sizes.push_back(16);
	default_powers.push_back(4);
	
	float minv = -1.0;
	float maxv = 1.0;
	std::vector<uint64_t> seeds;
	std::vector<uint64_t> default_seeds;
	default_seeds.push_back(0xDEADBEEF);
    for(i = 1; i < argc; i++)
    {
            // This is an option.
        if(argv[i][0]=='-')
        {
            switch(argv[i][1])
            {
                case 'o':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        stat_file = argv[i+1];
                    break;
                case 'r':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        reps = atoi(argv[i+1]);
                    break;
                case 's':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            size = atoi(argv[i+1]);
	                        sizes.push_back(size);
                        }
                        else
                            break;
                    }
                    break;
                case 'p':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            power = atoi(argv[i+1]);
	                        powers.push_back(power);
                        }
                        else
                            break;
                    }
                    break;
                case 'M':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            mhz = atoi(argv[i+1]);
	                        mhz_s.push_back(mhz);
                        }
                        else
                            break;
                    }
                    break;
                case 'f':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            functions.push_back(std::string(argv[i+1]));
                        }
                    else
                        break;
                    }
                    break;
                case 'i':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        iterations = atoi(argv[i+1]);
                    break;
                case 'h':
                    std::cout << "-s set the size of the matrix to multiply.\n-p set the power to raise it to.\n-f what functions to run.\n-d sets the random seed.\n-o sets where statistics should go.\n-i sets the number of iterations.\n-v compares the result with the reference solution.\n";
                    break;
                case 'v':
		            verify = 1;
                    break;
                }
            }
        }
	if(stat_file==NULL)
	    stat_file = default_filename;

	if (std::find(functions.begin(), functions.end(), "ALL") != functions.end()) {
		functions.clear();
		for(auto & f : function_map::get()) {
			functions.push_back(f.first);
		}
	}
	
	for(auto & function : functions) {
		auto t= function_map::get().find(function);
		if (t == function_map::get().end()) {
			std::cerr << "Unknown function: " << function <<"\n";
			exit(1);
		}
		std::cerr << "Gonna run " << function << "\n";
	}
	if(sizes.size()==0)
	    sizes = default_sizes;
	if(seeds.size()==0)
	    seeds = default_seeds;
	if(functions.size()==0)
	    functions = default_functions;
	if(powers.size()==0)
	    powers = default_powers;
	if(verify == 1)
            sprintf(header,"size,power,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses,correctness");
	else
	   sprintf(header,"size,power,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses");
    perfstats_print_header(stat_file, header);
     
	for(auto & seed: seeds ) {
		for(auto & size:sizes) {
			for(auto & power: powers ) {
				for(auto & function : functions) {
					tensor_t<ELEMENT_TYPE> dst(size,size);
					tensor_t<ELEMENT_TYPE> A(size,size);
					randomize(A, seed, minv, maxv);
							
					std::cerr << "Running: " << function << "\n";
					function_spec_t f_spec = function_map::get()[function];
					auto fut = reinterpret_cast<matexp_impl>(f_spec.second);
					sprintf(preamble, "%lu,%d,%s,",size,power,function.c_str());
					perfstats_init();
					unsigned long long* garbage = perfstats_enable();
					fut(dst, A, power, seed, iterations);
					perfstats_disable(garbage);
					if(verify)
					{
						tensor_t<ELEMENT_TYPE>::diff_prints_deltas = true;
						if(function.find("bench_solution") != std::string::npos)
						{
							function_spec_t t = function_map::get()[std::string("bench_reference")];
							auto verify_fut = reinterpret_cast<matexp_impl>(t.second);
							tensor_t<ELEMENT_TYPE> v(size,size);
							verify_fut(v, A, power, seed, 1);
							if(v == dst)
							{
								std::cerr << "Passed!\n";
								sprintf(epilogue,",passed\n");
							}
							else
							{
								std::cerr << "Check:\n" << A << "\nRAISED TO THE " << power << " SHOULD BE  \n" << v << "\nYOUR CODE GOT\n" << dst<< "\n";
								sprintf(epilogue,",failed\n");
							}
						}
						
						else if(function.find("matexp_solution_c") != std::string::npos)
						{
							function_spec_t t = function_map::get()[std::string("matexp_reference_c")];
							auto verify_fut = reinterpret_cast<matexp_impl>(t.second);
							tensor_t<ELEMENT_TYPE> v(size,size);							
							verify_fut(v, A, power, seed, 1);
							if(v == dst)
							{
								std::cerr << "Passed!!\n";
								sprintf(epilogue,",1\n");
							}
							else
							{
								std::cerr << diff(v,A);
							//	std::cerr << "Check:\n" << A << "\nRAISED TO THE " << power << " SHOULD BE  \n" << v << "\nYOUR CODE GOT\n" << dst<< "\n";
								sprintf(epilogue,",-1\n");
							}
						}
						else
						    sprintf(epilogue,",0\n");
					}
					else
						sprintf(epilogue,"\n");
					perfstats_print(preamble, stat_file, epilogue);
					perfstats_deinit();
					std::cerr << "Done execution: " << function << "\n";
				}
			}
		}
	}
	return 0;
}
