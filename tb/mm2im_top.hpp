// Jude: Created MM2IMv2
// Top-level wrapper for the MM2IM testbench. The configuration (geometry, types,
// weights, input, expected output) comes from the golden.h that
// tb/gen_mm2im_golden.py writes; put its directory on the include path.
#ifndef MM2IM_TOP_HPP
#define MM2IM_TOP_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "golden.h"

#ifndef MM2IM_SKIP
#define MM2IM_SKIP 1
#endif
// MM2IM_STREAM=1: mm2im_stream(), weights streamed per pixel (SKIP is then off)
#ifndef MM2IM_STREAM
#define MM2IM_STREAM 0
#endif

#if MM2IM_STREAM
void mm2im_top(
	hls::stream<hls::vector<TW, PE*SIMD>> &wgt,
	hls::stream<hls::vector<TI, SIMD>>    &src,
	hls::stream<hls::vector<TO, PE>>      &dst
);
#else
void mm2im_top(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, PE>>   &dst
);
#endif

#endif
