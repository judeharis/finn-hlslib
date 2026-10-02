// Jude: Created
/******************************************************************************
 *  MM2IM transposed convolution (input-stationary scatter), streaming NHWC.
 *
 *  Each input pixel is multiplied by every kernel tap and the products are
 *  scattered into an accumulator holding the output rows still receiving
 *  contributions (the MM2IM / col2im scheme of the SECDA TCONV accelerator,
 *  as a templated HLS function in the style of deconv.hpp).
 *
 *  Unlike deconv.hpp (gather), any K, S and P < K are supported, including
 *  K % S != 0 and K < S.
 *
 *  Per input row ih:
 *   - compute: for each pixel, taps (ky,kx) -> cf -> sf, one step per cycle.
 *     PE x SIMD MACs go into a per-PE register; every SF steps it is added
 *     into acc (read-modify-write). The pixel's SF input words are read during
 *     its first tap pass and kept in ibuf.
 *   - drain: output band ih (S rows) is complete; it is emitted and cleared.
 *  After the last row the remaining KB-1 bands are drained.
 *
 *  SKIP: visit only taps whose output survives the crop (fewer cycles). With
 *  SKIP=false every tap is visited and cropped products are discarded, so the
 *  weight sequence is identical for every pixel (needed for streamed weights).
 *
 *  Read-after-write on acc: the same address recurs after at least a few RMW
 *  events; the last L written (address, value) pairs are forwarded, so acc can
 *  be declared free of inter-iteration dependences.
 *
 *  Weights: kernel[t][pe][simd] with t = ((ky*K + kx)*CF + cf)*SF + sf and
 *  output channel co = cf*PE + pe, input channel ci = sf*SIMD + simd. From a
 *  FINN weight tensor W[CO][K][K][CI] (numpy):
 *    W.reshape(CF,PE,K,K,SF,SIMD).transpose(2,3,0,4,1,5).reshape(-1,PE,SIMD)
 ******************************************************************************/
#ifndef MM2IM_HPP
#define MM2IM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "mm2im_sched.hpp"

/**
 * Emit and clear the kept rows of output band b, held in accumulator bank `bank`.
 */
template<
	unsigned  S, unsigned  WO, unsigned  CF, unsigned  DEPTH,
	size_t  PE, typename  TA, typename  TO, typename  G
>
void mm2im_drain(
	TA  (&acc)[DEPTH][PE],
	unsigned const  bank,
	unsigned const  b,
	hls::stream<hls::vector<TO, PE>> &dst
) {
#pragma HLS inline
	unsigned const  n = G::rows_kept(b) * WO * CF;
	unsigned  rr = G::row_lo(b);
	unsigned  ow = 0;
	unsigned  cf = 0;
	for(unsigned  it = 0; it < n; it++) {
#pragma HLS pipeline II=1 style=flp
#pragma HLS loop_tripcount min=0 max=S*WO*CF
		unsigned const  addr = ((bank*S + rr)*WO + ow)*CF + cf;
		hls::vector<TO, PE>  y;
		for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
			y[pe] = TO(acc[addr][pe]);
			acc[addr][pe] = TA(0);
		}
		dst.write(y);

		if(cf != CF-1)  cf++;
		else {
			cf = 0;
			if(ow != WO-1)  ow++;
			else {
				ow = 0;
				rr++;
			}
		}
	}
} // mm2im_drain()

template<
	unsigned  K,	// kernel size
	unsigned  S,	// stride
	unsigned  P,	// (de)padding, P < K
	unsigned  H,	// input height
	unsigned  W,	// input width
	unsigned  CO,	// output channels
	unsigned  CI,	// input channels
	size_t    PE,	// output channels in parallel
	size_t    SIMD,	// input channels in parallel
	typename  TW,	// weight type
	typename  TI,	// input type
	typename  TO,	// output type
	typename  TA = TO,	// accumulator type
	bool      SKIP = true,	// skip taps that land in the crop
	unsigned  L = 4		// RMW forwarding window (RMW events)
>
void mm2im(
	TW const (&kernel)[K*K*(CO/PE)*(CI/SIMD)][PE][SIMD],
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, PE>>   &dst
) {
	static_assert(CO%PE   == 0, "PE parallelism must divide output channel count.");
	static_assert(CI%SIMD == 0, "SIMD parallelism must divide input channel count.");
	static_assert(L > 0, "Forwarding window must not be empty.");

	using  G = mm2im_geom<K, S, P, H, W>;
	constexpr unsigned  CF = CO/PE;
	constexpr unsigned  SF = CI/SIMD;
	constexpr unsigned  NB = G::KB;	// accumulator banks, one per live band
	constexpr unsigned  HO = G::HO;
	constexpr unsigned  WO = G::WO;
	constexpr unsigned  TX = G::taps_x(SKIP);
	constexpr unsigned  DEPTH = NB*S*WO*CF;

#pragma HLS array_partition variable=kernel complete dim=2
#pragma HLS array_partition variable=kernel complete dim=3

	// Accumulator: bank (band mod NB), row in band, output column, cf; PE lanes.
	// Static and zero-initialised; every drained entry is cleared again.
	static TA  acc[DEPTH][PE];
#pragma HLS array_partition variable=acc complete dim=2

	hls::vector<TI, SIMD>  ibuf[SF];

	unsigned  bank0 = 0;	// bank of band ih
	for(unsigned  ih = 0; ih < H; ih++) {
		unsigned const  ky_lo = SKIP? G::tap_lo(ih)     : 0;
		unsigned const  ky_hi = SKIP? G::tap_hi(ih, HO) : K;
		unsigned const  n = (ky_hi - ky_lo) * TX * CF * SF;

		unsigned  iw = 0;
		unsigned  kx_lo = SKIP? G::tap_lo(0)     : 0;
		unsigned  kx_hi = SKIP? G::tap_hi(0, WO) : K;
		unsigned  ky = ky_lo;
		unsigned  kx = kx_lo;
		unsigned  cf = 0;
		unsigned  sf = 0;

		TA  psum[PE];
#pragma HLS array_partition variable=psum complete
		unsigned  fwd_addr[L];
		TA        fwd_val[L][PE];
		bool      fwd_ok[L];
#pragma HLS array_partition variable=fwd_addr complete
#pragma HLS array_partition variable=fwd_val complete dim=0
#pragma HLS array_partition variable=fwd_ok complete
		for(unsigned  j = 0; j < L; j++) {
#pragma HLS unroll
			fwd_ok[j] = false;
		}

		for(unsigned  it = 0; it < n; it++) {
#pragma HLS pipeline II=1 style=flp
#pragma HLS dependence variable=acc inter false
#pragma HLS loop_tripcount min=1 max=K*TX*CF*SF

			// Input: read during the pixel's first tap pass, replayed afterwards.
			bool const  first = (ky == ky_lo) && (kx == kx_lo) && (cf == 0);
			hls::vector<TI, SIMD>  x;
			if(first) {
				x = src.read();
				ibuf[sf] = x;
			}
			else  x = ibuf[sf];

			// PE x SIMD MACs into the per-PE partial sum.
			unsigned const  t = ((ky*K + kx)*CF + cf)*SF + sf;
			for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
				TA  p = (sf == 0)? TA(0) : psum[pe];
				for(unsigned  s = 0; s < SIMD; s++) {
#pragma HLS unroll
					p += kernel[t][pe][s] * x[s];
				}
				psum[pe] = p;
			}

			// Scatter: add the finished partial sum into its output position.
			if(sf == SF-1) {
				unsigned const  r = ih*S + ky;	// full output row
				unsigned const  c = iw*S + kx;	// full output column
				bool const  keep = SKIP || ((r >= P) && (r < P+HO) && (c >= P) && (c < P+WO));
				if(keep) {
					unsigned  bank = bank0 + ky/S;
					if(bank >= NB)  bank -= NB;
					unsigned const  addr = ((bank*S + ky%S)*WO + (c - P))*CF + cf;

					TA  v[PE];
#pragma HLS array_partition variable=v complete
					for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
						v[pe] = acc[addr][pe];
					}
					for(unsigned  j = L; j-- > 0;) {	// oldest first, newest wins
#pragma HLS unroll
						if(fwd_ok[j] && (fwd_addr[j] == addr)) {
							for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
								v[pe] = fwd_val[j][pe];
							}
						}
					}
					for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
						v[pe] += psum[pe];
						acc[addr][pe] = v[pe];
					}
					for(unsigned  j = L-1; j > 0; j--) {
#pragma HLS unroll
						fwd_ok[j]   = fwd_ok[j-1];
						fwd_addr[j] = fwd_addr[j-1];
						for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
							fwd_val[j][pe] = fwd_val[j-1][pe];
						}
					}
					fwd_ok[0]   = true;
					fwd_addr[0] = addr;
					for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS unroll
						fwd_val[0][pe] = v[pe];
					}
				}
			}

			// Advance sf -> cf -> kx -> ky -> iw.
			if(sf != SF-1)  sf++;
			else {
				sf = 0;
				if(cf != CF-1)  cf++;
				else {
					cf = 0;
					if(kx != kx_hi-1)  kx++;
					else if(ky != ky_hi-1) {
						ky++;
						kx = kx_lo;
					}
					else {
						ky = ky_lo;
						iw++;
						kx_lo = SKIP? G::tap_lo(iw)     : 0;
						kx_hi = SKIP? G::tap_hi(iw, WO) : K;
						kx = kx_lo;
					}
				}
			}
		}

		// Band ih is complete: emit it.
		mm2im_drain<S, WO, CF, DEPTH, PE, TA, TO, G>(acc, bank0, ih, dst);
		bank0 = (bank0 == NB-1)? 0 : bank0 + 1;
	}

	// Bands below the last input row.
	for(unsigned  b = H; b < G::BANDS; b++) {
		mm2im_drain<S, WO, CF, DEPTH, PE, TA, TO, G>(acc, bank0, b, dst);
		bank0 = (bank0 == NB-1)? 0 : bank0 + 1;
	}

} // mm2im()

#endif
