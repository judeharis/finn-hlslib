// Jude: Created MM2IMv2
#include "mm2im_top.hpp"
#include "mm2im.hpp"

#if MM2IM_STREAM
void mm2im_top(
	hls::stream<hls::vector<TW, PE*SIMD>> &wgt,
	hls::stream<hls::vector<TI, SIMD>>    &src,
	hls::stream<hls::vector<TO, PE>>      &dst
) {
#pragma HLS interface axis port=wgt
#pragma HLS interface axis port=src
#pragma HLS interface axis port=dst
#pragma HLS aggregate variable=wgt compact=bit
#pragma HLS aggregate variable=src compact=bit
#pragma HLS aggregate variable=dst compact=bit

	mm2im_stream<K, S, P, H, W, CO, CI, PE, SIMD, TW, TI, TO, TA>(wgt, src, dst);

} // mm2im_top()
#else
void mm2im_top(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, PE>>   &dst
) {
#pragma HLS interface axis port=src
#pragma HLS interface axis port=dst
#pragma HLS aggregate variable=src compact=bit
#pragma HLS aggregate variable=dst compact=bit

	mm2im<K, S, P, H, W, CO, CI, PE, SIMD, TW, TI, TO, TA, bool(MM2IM_SKIP)>(KERNEL, src, dst);

} // mm2im_top()
#endif
