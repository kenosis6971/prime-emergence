#!/usr/bin/env python3
"""
Prime Odometer — CLI

A generative, factor-free prime enumerator that advances n -> n+1 and updates
residues r[p] ≡ n (mod p) for discovered primes p. In extended mode, also tracks
residues modulo p^t to read off carry-depth valuations ν_p(n+1) without refactoring.

Usage examples:
  - Minimal enumeration up to 10_000:
      python prime_odometer.py --limit 10000 --print-primes

  - Extended mode with p^t stacks (t=3), show ν_p on composite ticks:
      python prime_odometer.py --limit 500 --extended --tpow 3 --print-nu

  - Quiet run with final stats:
      python prime_odometer.py --limit 100000 --stats
"""

from __future__ import annotations
import argparse
from math import isqrt
from typing import Dict, List, Tuple, Iterator


class PrimeOdometer:
    """
    Minimal Prime Odometer: factor-free primality decision and prime enumeration.
    Maintains r[p] ≡ n (mod p) for discovered primes p.
    """

    def __init__(self, start: int = 2):
        if start < 2:
            start = 2
        self.n: int = start
        self.P: List[int] = [2]              # discovered primes
        self.r: Dict[int, int] = {2: self.n % 2}  # residues mod p
        # stats
        self.ticks: int = 0
        self.primes_found: int = 1 if start == 2 else 0

    def tick(self) -> Tuple[int, bool]:
        """Advance to m = n+1, decide primality via residues up to sqrt(m)."""
        m = self.n + 1
        self.ticks += 1

        # 1) increment residues
        for p in self.P:
            self.r[p] = (self.r[p] + 1) % p

        # 2) test divisibility by p ≤ sqrt(m)
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

        # 3) m is prime → append, init residue
        self.P.append(m)
        self.r[m] = 0
        self.n = m
        self.primes_found += 1
        return (m, True)

    def enumerate(self, limit: int) -> Iterator[Tuple[int, bool]]:
        """Yield (m, is_prime) for each tick up to limit (inclusive)."""
        if self.n == 2:
            yield (2, True)
        while self.n < limit:
            yield self.tick()


class PrimeOdometerExt:
    """
    Extended Prime Odometer with optional ν_p(n+1) valuations via carry-depth.
    Tracks residues modulo p and p^t for small t, so we can read ν_p without refactoring m.
    """

    def __init__(self, start: int = 2, tpow_default: int = 2):
        if start < 2:
            start = 2
        self.n: int = start
        self.P: List[int] = [2]
        self.r: Dict[int, int] = {2: self.n % 2}
        self.R: Dict[int, Dict[int, int]] = {2: {}}
        self.tpow: Dict[int, int] = {2: max(2, tpow_default)}

        # init stacks for p=2
        for t in range(2, self.tpow[2] + 1):
            mod_pt = 2 ** t
            self.R[2][t] = self.n % mod_pt

        # stats
        self.ticks: int = 0
        self.primes_found: int = 1 if start == 2 else 0

    def _init_prime_stacks(self, p: int):
        """Initialize residue stacks for new prime p."""
        self.R[p] = {}
        if p not in self.tpow:
            self.tpow[p] = self.tpow.get(2, 2)  # use default depth
        for t in range(2, self.tpow[p] + 1):
            mod_pt = p ** t
            self.R[p][t] = self.n % mod_pt

    def tick(self) -> Tuple[int, bool, Dict[int, int]]:
        """
        Advance to m = n+1, update residues modulo p and p^t, decide primality,
        and return ν_p(m) valuations (possibly empty) for p ≤ sqrt(m).
        Returns (m, is_prime, nu_dict).
        """
        m = self.n + 1
        self.ticks += 1

        # 1) increment residues
        for p in self.P:
            self.r[p] = (self.r[p] + 1) % p
            # bump p^t stacks
            for t in range(2, self.tpow[p] + 1):
                mod_pt = p ** t
                self.R[p][t] = (self.R[p][t] + 1) % mod_pt

        # 2) divisibility and carry-depth
        bound = isqrt(m)
        nu: Dict[int, int] = {}
        is_composite = False

        for p in self.P:
            if p > bound:
                break
            if self.r[p] == 0:
                is_composite = True
                # detect deepest wrap
                tmax = 1
                for t in range(2, self.tpow[p] + 1):
                    if self.R[p][t] == 0:
                        tmax = t
                    else:
                        break
                nu[p] = tmax

        if is_composite:
            self.n = m
            return (m, False, nu)

        # 3) m is prime
        self.P.append(m)
        self.r[m] = 0
        self._init_prime_stacks(m)
        self.n = m
        self.primes_found += 1
        return (m, True, {m: 1})

    def enumerate(self, limit: int) -> Iterator[Tuple[int, bool, Dict[int, int]]]:
        """Yield (m, is_prime, nu_dict) for each tick up to limit (inclusive)."""
        if self.n == 2:
            yield (2, True, {2: 1})
        while self.n < limit:
            yield self.tick()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prime Odometer — generative, factor-free prime enumerator.")
    p.add_argument("--limit", type=int, required=True, help="Upper limit N (inclusive) to enumerate up to.")
    p.add_argument("--start", type=int, default=2, help="Start integer (default 2).")
    p.add_argument("--extended", action="store_true", help="Use extended mode with p^t residue stacks.")
    p.add_argument("--tpow", type=int, default=2, help="Depth t for p^t stacks in extended mode (default 2).")
    p.add_argument("--print-primes", action="store_true", help="Print primes as they are found.")
    p.add_argument("--print-nu", action="store_true", help="(Extended) Print ν_p on composite ticks.")
    p.add_argument("--stats", action="store_true", help="Print simple statistics at end.")
    return p


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.extended:
        od = PrimeOdometerExt(start=args.start, tpow_default=args.tpow)
        last_m = args.start
        for (m, is_prime, nu) in od.enumerate(args.limit):
            if is_prime and args.print_primes:
                print(m)
            elif (not is_prime) and args.print_nu and nu:
                # print composite with valuations ν_p(m)
                parts = [f"{p}^{nu[p]}" if nu[p] > 1 else f"{p}" for p in sorted(nu)]
                print(f"{m} = {' * '.join(parts)} * (cofactor)")
            last_m = m
        if args.stats:
            print("--- stats ---")
            print(f"ticks          : {od.ticks}")
            print(f"primes found   : {od.primes_found}")
            print(f"largest prime  : {od.P[-1]}")
            print(f"basis size |P| : {len(od.P)}")
            print(f"ended at n     : {od.n}")

    else:
        od = PrimeOdometer(start=args.start)
        last_m = args.start
        for (m, is_prime) in od.enumerate(args.limit):
            if is_prime and args.print_primes:
                print(m)
            last_m = m
        if args.stats:
            print("--- stats ---")
            print(f"ticks          : {od.ticks}")
            print(f"primes found   : {od.primes_found}")
            print(f"largest prime  : {od.P[-1]}")
            print(f"basis size |P| : {len(od.P)}")
            print(f"ended at n     : {od.n}")

if __name__ == "__main__":
    main()
