/* CCT-L3 native decoder.
 *
 * Reassembles BF16 values from the three L3 streams:
 *
 *   sign : n bits   packed MSB-first   (already inflated by zstd in Python)
 *   exp  : n bytes                     (already inflated by zstd in Python)
 *   mant : 7n bits  packed MSB-first   (raw in the payload)
 *
 * This is the hot part of the L3 decode: in numpy it costs ~100 ms for
 * 4 MB of weights; here it is memory-bound.
 *
 * Output is written byte by byte (little-endian BF16 layout), so the
 * routine is endian-independent. All input bounds are checked against the
 * lengths supplied by the caller: no padding is required.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(_WIN32)
#define CCT_EXPORT __declspec(dllexport)
#else
#define CCT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

CCT_EXPORT int l3_unpack_bf16(const uint8_t *sign_bits, int64_t sign_len,
                              const uint8_t *exp, int64_t exp_len,
                              const uint8_t *mant_bits, int64_t mant_len,
                              int64_t n,
                              uint8_t *out /* 2*n bytes */)
{
    if (n < 0 || sign_len < 0 || exp_len < 0 || mant_len < 0) return -1;
    if (n == 0) return 0;
    if (sign_len < (n + 7) / 8) return -2;
    if (exp_len < n) return -3;
    if (mant_len < (7 * n + 7) / 8) return -4;

    int64_t i = 0;

    /* Fast path: 8 values consume exactly 56 mantissa bits == 7 bytes. */
    for (; i + 8 <= n && ((i * 7) >> 3) + 8 <= mant_len; i += 8) {
        uint64_t w = 0;
        int64_t byte = (i * 7) >> 3;
        memcpy(&w, mant_bits + byte, sizeof(w));

        /* Big-endian bit order: the first 56 bits live in the high part. */
#if !defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_BIG_ENDIAN__)
        w = __builtin_bswap64(w);
#endif

        uint8_t sbyte = sign_bits[i >> 3];
        uint8_t *o = out + 2 * i;
        const uint8_t *e = exp + i;

        for (int k = 0; k < 8; k++) {
            /* stream bit 7k is word bit 63-7k; field ends at 57-7k */
            uint16_t m = (uint16_t)((w >> (57 - 7 * k)) & 0x7Fu);
            uint8_t s = (uint8_t)((sbyte >> (7 - k)) & 1u);
            uint8_t ev = e[k];
            /* BF16 little-endian: low byte = (exp&1)<<7 | mant,
                                high byte = sign<<7 | exp>>1 */
            o[2 * k]     = (uint8_t)(((ev & 1u) << 7) | m);
            o[2 * k + 1] = (uint8_t)((s << 7) | (ev >> 1));
        }
    }

    /* Tail (also covers reads that would run past mant_len). */
    for (; i < n; i++) {
        uint8_t s = (uint8_t)((sign_bits[i >> 3] >> (7 - (i & 7))) & 1u);
        int64_t bit = i * 7;
        int64_t byte = bit >> 3;
        int shift = 24 - (int)(bit & 7) - 7;
        uint32_t w = 0;
        if (byte + 2 < mant_len) {
            w = ((uint32_t)mant_bits[byte] << 16)
              | ((uint32_t)mant_bits[byte + 1] << 8)
              | ((uint32_t)mant_bits[byte + 2]);
        } else {
            for (int k = 0; k < 3; k++) {
                int64_t idx = byte + k;
                if (idx < mant_len) w |= ((uint32_t)mant_bits[idx]) << (16 - 8 * k);
            }
        }
        uint16_t m = (uint16_t)((w >> shift) & 0x7Fu);
        uint8_t ev = exp[i];
        out[2 * i]     = (uint8_t)(((ev & 1u) << 7) | m);
        out[2 * i + 1] = (uint8_t)((s << 7) | (ev >> 1));
    }

    return 0;
}

/* ABI marker so the loader can reject a stale/incompatible build. */
CCT_EXPORT int l3_native_abi(void) { return 4; }

#ifdef __cplusplus
}
#endif
