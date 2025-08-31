# prime_odometer_min.py

from math import isqrt
from typing import Dict, List, Tuple, Iterator

class PrimeOdometer:
    """
    Minimal Prime Odometer: factor-free primality decision and prime enumeration.
    Maintains residues r[p] ≡ n (mod p) for discovered primes p, with p^2 ≥ n.
    """

    def __init__(self, start: int = 2):
        if start < 2:
            start = 2
        self.n: int = start
        # seed with first prime
        self.P: List[int] = [2]
        # r[p] ≡ n (mod p)
        self.r: Dict[int, int] = {2: self.n % 2}

    def _ensure_base(self):
        """
        Ensure our prime basis includes all primes up to floor(sqrt(n)).
        If we're just enumerating forward from 2, this is maintained incrementally
        by appending new primes when they appear; no extra work is needed here.
        """
        pass  # kept for symmetry/clarity; basis grows as we discover new primes

    def tick(self) -> Tuple[int, bool]:
        """
        Advance to m = n+1, update residues, decide primality by residues for p ≤ sqrt(m).
        Returns (m, is_prime).
        """
        m = self.n + 1

        # 1) increment residues mod p for all discovered primes
        for p in self.P:
            self.r[p] = (self.r[p] + 1) % p

        # 2) test divisibility by any discovered prime up to sqrt(m)
        bound = isqrt(m)
        is_composite = False
        for p in self.P:
            if p > bound:
                break
            if self.r[p] == 0:
                is_composite = True
                break

        if is_composite:
            self.n = m
            return (m, False)

        # 3) no small prime divides m → m is prime. Append and init its residue.
        self.P.append(m)
        self.r[m] = 0  # since m ≡ 0 (mod m)
        self.n = m
        return (m, True)

    def primes_up_to(self, N: int) -> Iterator[int]:
        """Yield primes from current position up to N (inclusive)."""
        # yield any primes already at/behind current n
        if self.n == 2:
            yield 2
        while self.n < N:
            m, is_prime = self.tick()
            if is_prime:
                yield m

if __name__ == "__main__":
    od = PrimeOdometer(start=2)
    # Demo: primes up to 200
    print(list(od.primes_up_to(200)))
