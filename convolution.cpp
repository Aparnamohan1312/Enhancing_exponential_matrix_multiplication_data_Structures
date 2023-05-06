
#include"pin_tags.h"
#include"CNN/tensor_t.hpp"
#include"function_map.hpp"
#include<cstdint>
#include<cassert>

//START
extern "C"
void do_convolution(const tensor_t<uint32_t> & source,
                    const tensor_t<uint32_t> & kernel,
                    tensor_t<uint32_t> & target, int32_t tile_size) {
	for(register int32_t i = 0; i < target.size.x; i++) {
		for(register int32_t j = 0; j < kernel.size.x; j++) {
			target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
		}
	}
}

extern "C"
uint64_t* convolution(uint64_t * source_space, uint64_t source_size,
                      uint64_t * kernel_space, uint64_t kernel_size, 
                      uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	
	// Here's the the key part:
	do_convolution(source, kernel, target, tile_size);
  
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;
}

FUNCTION(convolution, convolution);
//END


extern "C"
void do_convolution_new_loop(const tensor_t<uint32_t> & source,
			     const tensor_t<uint32_t> & kernel,
			     tensor_t<uint32_t> & target, int32_t tile_size) {

	for(int32_t i = 0; i < target.size.x; i++) {
		for(int32_t jj = 0; jj < kernel.size.x; jj += tile_size) { // We create a new loop variable jj that advanced one chunk at a time.
			for(int32_t j = jj; j < kernel.size.x && j < jj + tile_size; j++) { // We iterate over the chunk.  The more complicated termination 
				// condition keeps ups from running off the end of the arry
				target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
			}
		}
	}
  

}

extern "C"
uint64_t* convolution_new_loop(uint64_t * source_space, uint64_t source_size,
			       uint64_t * kernel_space, uint64_t kernel_size, 
			       uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_new_loop(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;
}

FUNCTION(convolution, convolution_new_loop);


extern "C"
void do_convolution_split(const tensor_t<uint32_t> & source,
			  const tensor_t<uint32_t> & kernel,
			  tensor_t<uint32_t> & target, int32_t tile_size) {

	for(int32_t i = 0; i < target.size.x; i++) {
		for(int32_t jj = 0; jj < kernel.size.x; jj += 2048) { // We create a new loop variable jj that advanced one chunk at a time.
			for(int32_t j = jj; j < kernel.size.x && j < jj + 2048; j++) { // We iterate over the chunk.  The more complicated termination 
				// condition keeps ups from running off the end of the arry
				target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
			}
		}
	}
  

}

extern "C"
uint64_t* convolution_split(uint64_t * source_space, uint64_t source_size,
			    uint64_t * kernel_space, uint64_t kernel_size, 
			    uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_split(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;
}

FUNCTION(convolution, convolution_split);



extern "C"
void do_convolution_tiled(const tensor_t<uint32_t> & source,
			  const tensor_t<uint32_t> & kernel,
			  tensor_t<uint32_t> & target, int32_t tile_size) {

	for(int32_t jj = 0; jj < kernel.size.x; jj += tile_size) {  // Move the jj chunk loop outside
		for(int32_t i = 0; i < target.size.x; i++) {
			for(int32_t j = jj; j < kernel.size.x && j < jj + tile_size; j++) {
				target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
			}
		}
	}
}

extern "C"
uint64_t* convolution_tiled(uint64_t * source_space, uint64_t source_size,
			    uint64_t * kernel_space, uint64_t kernel_size, 
			    uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_tiled(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;  
}

FUNCTION(convolution, convolution_tiled);

extern "C"
void  __attribute__((optimize("unroll-loops"))) do_convolution_tiled_unrolled(const tensor_t<uint32_t> & source,
									      const tensor_t<uint32_t> & kernel,
									      tensor_t<uint32_t> & target, int32_t tile_size) {

	for(int32_t jj = 0; jj < kernel.size.x; jj += tile_size) {  // Move the jj chunk loop outside
		for(int32_t i = 0; i < target.size.x; i++) {
			for(int32_t j = jj; j < kernel.size.x && j < jj + tile_size; j++) {
				target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
			}
		}
	}

}



extern "C"
uint64_t* convolution_tiled_unrolled(uint64_t * source_space, uint64_t source_size,
				     uint64_t * kernel_space, uint64_t kernel_size, 
				     uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_tiled_unrolled(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;  
}

FUNCTION(convolution, convolution_tiled_unrolled);


extern "C"
void __attribute__((optimize("unroll-loops"))) do_convolution_tiled_split(const tensor_t<uint32_t> & source, 
									  const tensor_t<uint32_t> & kernel,
									  tensor_t<uint32_t> & target, int32_t tile_size) {

	int32_t real_tile_size = tile_size/8 * 8; // this clears the low 3 bits.  Check the assembly!
	assert(tile_size>=8);

	for(int32_t jj = 0; jj < kernel.size.x; jj += real_tile_size) {  // Move the jj chunk loop outside
		for(int32_t i = 0; i < target.size.x; i++) {
			if (jj + real_tile_size > kernel.size.x) {
				for(int32_t j = jj; j < kernel.size.x; j++) {
					target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
				} 
			} else {
				for(int32_t j = jj; j < jj + real_tile_size; j++) {
					target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
				}
			}
		}
	}
	

}

extern "C"
uint64_t* convolution_tiled_split(uint64_t * source_space, uint64_t source_size,
				  uint64_t * kernel_space, uint64_t kernel_size, 
				  uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_tiled_split(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;
}

FUNCTION(convolution, convolution_tiled_split);

extern "C"
void __attribute__((optimize("unroll-loops"))) do_convolution_tiled_fixed_tile(const tensor_t<uint32_t> & source, 
									  const tensor_t<uint32_t> & kernel,
									  tensor_t<uint32_t> & target, int32_t tile_size) {

#define real_tile_size 1024
//	int32_t real_tile_size = tile_size/8 * 8; // this clears the low 3 bits.  Check the assembly!

	
	for(int32_t jj = 0; jj < kernel.size.x; jj += real_tile_size) {  // Move the jj chunk loop outside
		for(int32_t i = 0; i < target.size.x; i++) {
			if (jj + real_tile_size > kernel.size.x) {
				for(int32_t j = jj; j < kernel.size.x; j++) {
					target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
				} 
			} else {
				for(int32_t j = jj; j < jj + real_tile_size; j++) {
					target.get(i,0,0,0) += source.get(i + j,0,0,0) * kernel.get(j,0,0,0);
				}
			}
		}
	}
	

}

extern "C"
uint64_t* convolution_tiled_fixed_tile(uint64_t * source_space, uint64_t source_size,
				  uint64_t * kernel_space, uint64_t kernel_size, 
				  uint64_t * target_space, uint64_t _target_size, int32_t tile_size) {
	tensor_t<uint32_t> source(source_size,1,1,1, (uint32_t *)source_space);
	tensor_t<uint32_t> kernel(kernel_size,1,1,1, (uint32_t *)kernel_space);
	uint64_t target_size = source_size - kernel_size;
	tensor_t<uint32_t> target(target_size,1,1,1, (uint32_t *)target_space);
	TAG_START("source", source.data, &source.as_vector(source.element_count()), true);
	TAG_START("kernel", kernel.data, &kernel.as_vector(kernel.element_count()), true);
	TAG_START("target", target.data, &target.as_vector(target.element_count()), true);

	// Here's the the key part:
	do_convolution_tiled_fixed_tile(source, kernel, target, tile_size);
    
	TAG_STOP("source");
	TAG_STOP("kernel");
	TAG_STOP("target");
	return target_space;
}

FUNCTION(convolution, convolution_tiled_fixed_tile);


//-O3 -funroll-all-loops
