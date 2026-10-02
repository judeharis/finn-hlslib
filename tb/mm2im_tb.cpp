// Jude: Created
// MM2IM testbench: feeds golden.h's input for FRAMES frames (the accumulator must
// come back clean after every frame) and compares every output word with the
// golden, which tb/gen_mm2im_golden.py checked against numpy, PyTorch and the
// kernel model. Exit code 0 = bit-exact.
#include "mm2im_top.hpp"

#include <iostream>

#ifndef FRAMES
#define FRAMES 2
#endif

int main() {
	constexpr unsigned  SF = CI/SIMD;
	constexpr unsigned  CF = CO/PE;
	hls::stream<hls::vector<TI, SIMD>>  src;
	hls::stream<hls::vector<TO, PE>>    dst;

	unsigned  errors = 0;
	for(unsigned  frame = 0; frame < FRAMES; frame++) {
		for(unsigned  pix = 0; pix < H*W; pix++) {
			for(unsigned  sf = 0; sf < SF; sf++) {
				hls::vector<TI, SIMD>  x;
				for(unsigned  s = 0; s < SIMD; s++)  x[s] = TI(INPUT[pix*CI + sf*SIMD + s]);
				src.write(x);
			}
		}

		mm2im_top(src, dst);

		if(!src.empty()) {
			std::cerr << "frame " << frame << ": input not fully consumed" << std::endl;
			errors++;
		}
		for(unsigned  pix = 0; pix < HO*WO; pix++) {
			for(unsigned  cf = 0; cf < CF; cf++) {
				if(dst.empty()) {
					std::cerr << "frame " << frame << ": output ends early at pixel " << pix << std::endl;
					return  1;
				}
				auto const  y = dst.read();
				for(unsigned  pe = 0; pe < PE; pe++) {
					long long const  got = y[pe].to_int64();
					long long const  exp = EXPECTED[pix*CO + cf*PE + pe];
					if(got != exp) {
						if(errors < 10) {
							std::cerr << "frame " << frame << " oh " << pix/WO << " ow " << pix%WO
							          << " co " << cf*PE + pe << ": got " << got << " expected " << exp << std::endl;
						}
						errors++;
					}
				}
			}
		}
		if(!dst.empty()) {
			std::cerr << "frame " << frame << ": extra output words" << std::endl;
			errors++;
		}
	}

	std::cout << (errors? "FAIL" : "PASS") << " K=" << K << " S=" << S << " P=" << P
	          << " H=" << H << " W=" << W << " CI=" << CI << " CO=" << CO
	          << " PE=" << PE << " SIMD=" << SIMD << " SKIP=" << MM2IM_SKIP
	          << " frames=" << FRAMES << " errors=" << errors
	          << " iterations/frame=" << (MM2IM_SKIP? ITER_SKIP : ITER_NOSKIP) << std::endl;
	return  errors? 1 : 0;
}
