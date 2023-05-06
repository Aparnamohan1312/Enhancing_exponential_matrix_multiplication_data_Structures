#ifndef MATEXP_SOLUTION_INCLUDED
#define MATEXP_SOLUTION_INCLUDED
#include <cstdlib>
#include "archlab.h"
#include <unistd.h>
#include<cstdint>
#include"function_map.hpp"
#include"tensor_t.hpp"
#include <immintrin.h>
template<typename T>
void __attribute__((noinline)) mult_solution(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B) {
    const int size_bl = 32;//indicates block
    const int size_of_tl = 128;//indicates tiles
    const int tiles = C.size.x / size_of_tl;
    
    // Compute transpose of matrix B in parallel
    tensor_t<T> transposeB(B.size.y, B.size.x);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < B.size.x; i++) {
        for (int j = 0; j < B.size.y; j++) {
            transposeB.get(j, i) = B.get(i, j);
        }
    }
    
    // Loop tiling with unrolling and reordering
    #pragma omp parallel for schedule(static)
    for(int ac = 0; ac < tiles*tiles; ac++) {
        int a = ac / tiles;
        int c = ac % tiles;
        for(int bj = 0; bj < tiles*size_of_tl; bj += 4) {
            int b = bj / size_of_tl;
            int jj = bj % size_of_tl;
            for(int ii = a * size_of_tl; ii < std::min((a + 1) * size_of_tl, C.size.x); ii += 4) {
                T total[4][4] = {{0}};
                for(int k = c * size_of_tl; k < std::min((c + 1) * size_of_tl, A.size.y); k++) {
                    T aik = A.get(ii, k);
                    for(int i = 0; i < 4; i++) {
                        for(int j = 0; j < 4; j++) {
                            total[i][j] += aik * transposeB.get(jj+j, k);
                        }
                    }
                }
                for(int i = 0; i < 4; i++) {
                    for(int j = 0; j < 4; j++) {
                        if (ii+i < C.size.x && jj+j < C.size.y) {
                            C.get(ii+i, jj+j) += total[i][j];
                        }
                    }
                }
            }
        }
    }
}

//66.69
// template<typename T>
// void __attribute__((noinline)) mult_solution(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B) {
//     const int size_bl = 32;//indicates block
//     const int size_of_tl = 128;//indicates tiles
//     const int tiles = C.size.x / size_of_tl;
    
//     // Compute transpose of matrix B in parallel
//     tensor_t<T> transposeB(B.size.y, B.size.x);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < B.size.x; i++) {
//         for (int j = 0; j < B.size.y; j++) {
//             transposeB.get(j, i) = B.get(i, j);
//         }
//     }
    
//     // Loop tiling with unrolling and reordering
//     #pragma omp parallel for schedule(static)
//     for(int a = 0; a < tiles; a++) {
//         for(int c = 0; c < tiles; c++) {
//             for(int b = 0; b < tiles; b++) {
//                 for(int ii = a * size_of_tl; ii < std::min((a + 1) * size_of_tl, C.size.x); ii += 4) {
//                     for(int jj = b * size_of_tl; jj < std::min((b + 1) * size_of_tl, C.size.y); jj += 4) {
//                         T total[4][4] = {{0}};
//                         for(int k = c * size_of_tl; k < std::min((c + 1) * size_of_tl, A.size.y); k++) {
//                             T aik = A.get(ii, k);
//                             for(int i = 0; i < 4; i++) {
//                                 for(int j = 0; j < 4; j++) {
//                                     total[i][j] += aik * transposeB.get(jj+j, k);
//                                 }
//                             }
//                         }
//                         for(int i = 0; i < 4; i++) {
//                             for(int j = 0; j < 4; j++) {
//                                 if (ii+i < C.size.x && jj+j < C.size.y) {
//                                     C.get(ii+i, jj+j) += total[i][j];
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

//66.65
// template<typename T>
// void __attribute__((noinline)) mult_solution(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B) {
//     const int size_bl = 32;//indicates block
//     const int size_of_tl = 128;//indicates tiles
//     const int tiles = C.size.x / size_of_tl;
//     // Considering speedup,transpose of matrix B is computed here in parallel
//     tensor_t<T> transposeB(B.size.y, B.size.x);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < B.size.x; i++) {
//         for (int j = 0; j < B.size.y; j++) {
//             transposeB.get(j, i) = B.get(i, j);
//         }
//     }
//     //Here my aim was to implement parallelization along with collapse for the loops to improve speedup
//     #pragma omp parallel for collapse(4) schedule(static) reduction(+:C.data[:C.size.x*C.size.y])
//     for(int a = 0; a < tiles; a++) {
//         for(int b = 0; b < tiles; b++) {
//             for(int c = 0; c < tiles; c++) {
//                 for(int i = a * size_of_tl; i < std::min((a + 1) * size_of_tl, C.size.x); i++) {
//                     for(int j = b * size_of_tl; j < std::min((b + 1) * size_of_tl, C.size.y); j++) {
//                         T total = 0;
//                         for(int k = c * size_of_tl; k < std::min((c + 1) * size_of_tl, A.size.y); k++) {
//                             total += A.get(i, k) * transposeB.get(j, k);
//                         }
//                         C.get(i, j) += total;
//                     }
//                 }
//             }
//         }
//     }
// }

//gave 66.5
// template<typename T>
// void __attribute__((noinline)) mult_solution(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B) {
//     const int size_bl = 32;//indicates block
//     const int size_of_tl = 128;//indicates tiles
//     const int tiles = C.size.x / size_of_tl;
//     // Considering speedup,transpose of matrix B is computed here in parallel
//     tensor_t<T> transposeB(B.size.y, B.size.x);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < B.size.x; i++) {
//         for (int j = 0; j < B.size.y; j++) {
//             transposeB.get(j, i) = B.get(i, j);
//         }
//     }
//     //Here my aim was to implement parallelization along with collapse for the loops to improve speedup
//     #pragma omp parallel for collapse(3) schedule(static) reduction(+:C.data[:C.size.x*C.size.y])
//     for(int a = 0; a < tiles; a++) {
//         for(int b = 0; b < tiles; b++) {
//             for(int c = 0; c < tiles; c++) {
//                 const int si = a * size_of_tl;//start of i
//                 const int ei = std::min((a + 1) * size_of_tl, C.size.x);//end of i
//                 const int sj = b * size_of_tl;
//                 const int ej = std::min((b + 1) * size_of_tl, C.size.y);
//                 const int sk = c * size_of_tl;
//                 const int ek = std::min((c + 1) * size_of_tl, A.size.y);
//                 for(int i = si; i < ei; i++) {
//                     for(int j = sj; j < ej; j++) {
//                         T total = 0;
//                         for(int k = sk; k < ek; k++) {
//                             total += A.get(i, k) * transposeB.get(j, k);
//                         }
//                         C.get(i, j) += total;
//                     }
//                 }
//             }
//         }
//     }
// }
//first one
// template<typename T>
// void __attribute__((noinline)) mult_solution(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B) {
//     const int size_bl = 32;//indicates block
//     const int size_of_tl = 128;//indicates tiles
//     const int tiles = C.size.x / size_of_tl;
//     // Considering speedup,transpose of matrix B is computed here in parallel
//     tensor_t<T> transposeB(B.size.y, B.size.x);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < B.size.x; i++) {
//         for (int j = 0; j < B.size.y; j++) {
//             transposeB.get(j, i) = B.get(i, j);
//         }}
//     //Here my aim was to implement parallelization along with collapse for the loops to improve speedup
//     #pragma omp parallel for collapse(2) schedule(static)
//     for(int a = 0; a < tiles; a++) {
//         for(int b = 0; b < tiles; b++) {
//             for(int c = 0; c < tiles; c++) {
//                 const int si = a * size_of_tl;//start of i
//                 const int ei = std::min((a + 1) * size_of_tl, C.size.x);//end of i
//                 const int sj = b * size_of_tl;
//                 const int ej = std::min((b + 1) * size_of_tl, C.size.y);
//                 const int sk = c * size_of_tl;
//                 const int ek = std::min((c + 1) * size_of_tl, A.size.y);
//                 for(int i = si; i < ei; i++) {
//                     for(int j = sj; j < ej; j++) {
//                         T total = 0;
//                         for(int k = sk; k < ek; k++) {
//                             total += A.get(i, k) * transposeB.get(j, k);
//                         }
//                         C.get(i, j) += total;
//                     }
//                 }
//             }
//         }
//     }
// }
template<typename T>
void __attribute__((noinline)) matexp_solution(tensor_t<T> & dst, const tensor_t<T> & A, uint32_t power) {
    const int size_of_tl = 32;
    const int size_bl = 32;
   // Initialize identity matrix
    for(int32_t x = 0; x < dst.size.x; x++) {
        for(int32_t y = 0; y < dst.size.y; y++) {
            dst.get(x, y) = (x == y) ? 1 : 0; //I used ternary operator here which also helps me improve the speedup
        }
    }
   //from what was taught in the lecture, implementing the tiling approach
    for(uint32_t p = 0; p < power; p++) {
        tensor_t<T> B(dst);
        //parallelization will enhance speedup
        #pragma omp parallel for schedule(static)
        //Here I broke down the matrices into sub matrices and blocks which will help to compute faster
        for(int first = 0; first < dst.size.x; first += size_bl) {
            for(int second = 0; second < dst.size.y; second += size_bl) {
                for(int third = 0; third < A.size.y; third += size_bl) {
                    for(int i = first; i < std::min(first + size_bl, dst.size.x); i++) {
                        for(int j = second; j < std::min(second + size_bl, dst.size.y); j++) {
                            T total = 0;
                            for(int k = third; k < std::min(third + size_bl, A.size.y); k++) {
                                total += B.get(i, k) * A.get(k, j);
                            }
                            dst.get(i, j) = total;
                        }}
                }}
        }
    }
}
#endif