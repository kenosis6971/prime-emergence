# prime_odometer_ext.py

from math import isqrt
from typing import Dict, List, Tuple, Iterator, Optional

class PrimeOdometerExt:
    """
    Extended Prime Odometer with optional ν_p(m) valuations via carry-depth.
    Tracks residues modulo p and p^t for small t, enabling ν_p(m) detection without refactoring m.
    """

    def __init__(self, start: int = 2, tpow_default: int = 2):
        if start < 2:
            start = 2
        self.n: int = start
        self.P: List[int] = [2]
        # r[p] ≡ n (mod p)
        self.r: Dict[int, int] = {2: self.n % 2}
        # R[p][t] ≡ n (mod p^t) for t = 2..tpow[p]
        self.R: Dict[int, Dict[int, int]] = {2: {}}
        self.tpow: Dict[int, int] = {2: max(2, tpow_default)}  # track up to p^t, default small

        # Initialize stacks for p=2
        for t in range(2, self.tpow[2] + 1):
            mod_pt = 2 ** t
            self.R[2][t] = self.n % mod_pt

    def _init_prime_stacks(self, p: int):
        # initialize p^t stacks for a newly discovered prime
        self.R[p] = {}
        self.tpow[p] = self.tpow.get(p, 2)
        for t in range(2, self.tpow[p] + 1):
            mod_pt = p ** t
            self.R[p][t] = self.n % mod_pt  # current n residue

    def tick(self) -> Tuple[int, bool, Dict[int, int]]:
        """
        Advance to m = n+1, update residues (mod p and mod p^t), decide primality,
        and return ν_p(m) valuations (possibly empty) for discovered primes p ≤ sqrt(m).
        Returns (m, is_prime, nu_dict).
        """
        m = self.n + 1

        # 1) increment residues modulo p and p^t
        for p in self.P:
            self.r[p] = (self.r[p] + 1) % p
            # bump p^t stacks if we track them
            for t in range(2, self.tpow[p] + 1):
                mod_pt = p ** t
                self.R[p][t] = (self.R[p][t] + 1) % mod_pt

        # 2) detect divisibility and carry-depth ν_p(m)
        bound = isqrt(m)
        nu: Dict[int, int] = {}
        is_composite = False

        for p in self.P:
            if p > bound:
                break
            if self.r[p] == 0:
                is_composite = True
                # detect depth: largest t such that n ≡ -1 (mod p^t) → after +1, R[p][t] == 0
                tmax = 1
                for t in range(2, self.tpow[p] + 1):
                    if self.R[p][t] == 0:
                        tmax = t
                    else:
                        break
                nu[p] = tmax

        if is_composite:
            # NOTE: If you want the full σ(m), you can form the cofactor cf = m // prod(p**nu[p]).
            self.n = m
            return (m, False, nu)

        # 3) otherwise m is prime: append to basis, initialize residues and stacks
        self.P.append(m)
        self.r[m] = 0  # m ≡ 0 (mod m)
        self._init_prime_stacks(m)
        self.n = m
        return (m, True, {m: 1})

    def enumerate(self, N: int) -> Iterator[Tuple[int, bool, Dict[int, int]]]:
        """
        Yield (m, is_prime, nu_dict) for each tick up to N (inclusive).
        """
        if self.n == 2:
            yield (2, True, {2: 1})
        while self.n < N:
            yield self.tick()

if __name__ == "__main__":
    od = PrimeOdometerExt(start=2, tpow_default=3)
    out = []
    for (m, is_prime, nu) in od.enumerate(200):
        if is_prime:
            out.append(m)
    print(out)  # primes up to 200
