"""Binary arithmetic coder over integer frequency tables (after Nayuki's reference coder), 64-bit state.

The output bitstream b1 b2 b3 ... is the binary expansion of a real x in [0, 1) lying in the nested
symbol intervals. Reading past the end of the stream yields the pad bit (0 by default), so any bit list decodes. finish()
flushes enough bits that every continuation of a complete stream decodes to the same symbols.
"""
STATE_BITS = 64
FULL = 1 << STATE_BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
MASK = FULL - 1


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, MASK, 0, []

    def encode(self, lo, hi, total):
        """Narrow to the cumulative-frequency interval [lo, hi) out of total."""
        span = self.high - self.low + 1
        self.high = self.low + hi * span // total - 1
        self.low = self.low + lo * span // total
        while not (self.low ^ self.high) & HALF:
            bit = self.low >> (STATE_BITS - 1)
            self.bits.append(bit)
            self.bits.extend([bit ^ 1] * self.pending)
            self.pending = 0
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while self.low & ~self.high & QUARTER:
            self.pending += 1
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1

    def finish(self):
        # Pick a quarter-range inside [low, high]: [1/4, 1/2) if low < 1/4, else [1/2, 3/4).
        bit = 0 if self.low < QUARTER else 1
        self.bits.append(bit)
        self.bits.extend([bit ^ 1] * (self.pending + 1))
        return self.bits


class Decoder:
    def __init__(self, bits, pad=0):
        self.bits, self.pos, self.pad = [int(b) for b in bits], 0, pad
        self.low, self.high, self.code = 0, MASK, 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self._read()

    def _read(self):
        self.pos += 1
        return self.bits[self.pos - 1] if self.pos <= len(self.bits) else self.pad

    def target(self, total):
        """The cumulative frequency the next symbol's interval must contain."""
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def consume(self, lo, hi, total):
        span = self.high - self.low + 1
        self.high = self.low + hi * span // total - 1
        self.low = self.low + lo * span // total
        while not (self.low ^ self.high) & HALF:
            self.code = ((self.code << 1) & MASK) | self._read()
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while self.low & ~self.high & QUARTER:
            self.code = (self.code & HALF) | ((self.code << 1) & (MASK >> 1)) | self._read()
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1
