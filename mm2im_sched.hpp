// Jude: Created MM2IMv2
/******************************************************************************
 *  Geometry and tap schedule of the MM2IM transposed convolution.
 *
 *  Shared by the HLS kernel (mm2im.hpp) and the SECDA SystemC twin, so it has
 *  no HLS types and no pragmas. It is C++14 for Vitis HLS / g++ and also
 *  parses as C++98 for the SECDA hardware flow (Vivado HLS 2019.2 compiles
 *  SystemC as C++98): there the functions are plain inline functions and the
 *  static_asserts are dropped (MM2IM_CONSTEXPR, MM2IM_STATIC_ASSERT).
 *
 *  Transposed convolution, per dimension: input index i, kernel tap k and
 *  stride S produce full output index i*S + k (full size (N-1)*S + K). The
 *  padding P is then cropped from both ends, leaving O = (N-1)*S + K - 2*P.
 *  Output rows are grouped into bands of S rows: input row i writes the bands
 *  i .. i+KB-1 and band i is complete once input row i has been processed.
 ******************************************************************************/
#ifndef MM2IM_SCHED_HPP
#define MM2IM_SCHED_HPP

#if __cplusplus >= 201402L
#define MM2IM_CONSTEXPR constexpr
#define MM2IM_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define MM2IM_CONSTEXPR
#define MM2IM_STATIC_ASSERT(c, m)
#endif

template<
	unsigned  K,	// kernel size
	unsigned  S,	// stride
	unsigned  P,	// (de)padding cropped from each edge
	unsigned  H,	// input height
	unsigned  W		// input width
>
struct mm2im_geom {
	MM2IM_STATIC_ASSERT(K > 0 && S > 0 && H > 0 && W > 0, "Degenerate geometry.");
	MM2IM_STATIC_ASSERT(P < K, "Padding must be smaller than the kernel (P < K).");

	static const unsigned  HF = (H-1)*S + K;	// full output height
	static const unsigned  WF = (W-1)*S + K;	// full output width
	MM2IM_STATIC_ASSERT(2*P < HF && 2*P < WF, "Padding crops the whole output.");
	static const unsigned  HO = HF - 2*P;		// cropped output height
	static const unsigned  WO = WF - 2*P;		// cropped output width

	static const unsigned  KB = (K-1)/S + 1;	// bands touched by one input row
	static const unsigned  BANDS = H + KB - 1;	// bands holding any output row

	// First tap k of input index i whose output i*S+k survives the crop (>= P).
	static MM2IM_CONSTEXPR unsigned tap_lo(unsigned i) {
		return  i*S >= P? 0 : P - i*S;
	}
	// One past the last tap k of input index i with i*S+k < P+O.
	static MM2IM_CONSTEXPR unsigned tap_hi(unsigned i, unsigned O) {
		return  P + O <= i*S? 0 : (P + O - i*S >= K? K : P + O - i*S);
	}

	// Kept rows of band b, as offsets [row_lo, row_hi) within the band.
	static MM2IM_CONSTEXPR unsigned row_lo(unsigned b) {
		return  b*S >= P? 0 : (P - b*S >= S? S : P - b*S);
	}
	static MM2IM_CONSTEXPR unsigned row_hi(unsigned b) {
		return  P + HO <= b*S? 0 : (P + HO - b*S >= S? S : P + HO - b*S);
	}
	static MM2IM_CONSTEXPR unsigned rows_kept(unsigned b) {
		return  row_hi(b) > row_lo(b)? row_hi(b) - row_lo(b) : 0;
	}

	// Taps visited per input row along x, summed over all W columns.
	static MM2IM_CONSTEXPR unsigned taps_x(bool skip) {
		unsigned  n = 0;
		for(unsigned  i = 0; i < W; i++)  n += skip? tap_hi(i, WO) - tap_lo(i) : K;
		return  n;
	}
	// Taps visited along y for input row i.
	static MM2IM_CONSTEXPR unsigned taps_y(unsigned i, bool skip) {
		return  skip? tap_hi(i, HO) - tap_lo(i) : K;
	}

	// Loop iterations of one frame (compute + drain), excluding pipeline latency.
	static MM2IM_CONSTEXPR unsigned long long iterations(unsigned CF, unsigned SF, bool skip) {
		unsigned long long  n = 0;
		for(unsigned  i = 0; i < H; i++)  n += (unsigned long long)taps_y(i, skip) * taps_x(skip) * CF * SF;
		for(unsigned  b = 0; b < BANDS; b++)  n += (unsigned long long)rows_kept(b) * WO * CF;
		return  n;
	}
	// Expected cycles of one frame: iterations plus a pipeline fill/flush per input row.
	// Constants fitted to cosim/rtlsim (tb/gen_mm2im_golden.py CYCLE_ROW, CYCLE_CONST; keep
	// equal); accurate to max(3%, 6 cycles per input row).
	static MM2IM_CONSTEXPR unsigned long long exp_cycles(unsigned CF, unsigned SF, bool skip) {
		return  iterations(CF, SF, skip) + ((skip? 21ull : 17ull)*H > 34? (skip? 21ull : 17ull)*H - 34 : 0);
	}
};

#endif
