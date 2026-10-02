// Jude: Created
#include "mm2im_top.hpp"
#include "mm2im.hpp"

void mm2im_top(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, PE>>   &dst
) {
#pragma HLS interface axis port=src
#pragma HLS interface axis port=dst
#pragma HLS aggregate variable=src compact=bit
#pragma HLS aggregate variable=dst compact=bit

	mm2im<K, S, P, H, W, CO, CI, PE, SIMD, TW, TI, TO, TO, bool(MM2IM_SKIP)>(KERNEL, src, dst);

} // mm2im_top()
