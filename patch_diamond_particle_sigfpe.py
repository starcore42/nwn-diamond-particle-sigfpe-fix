#!/usr/bin/env python3
"""
Patch the NWN Diamond particle emitter SIGFPE bug.

The bug:

    Some Diamond particle code computes a random offset like this:

        divisor = (int)(0.5f * n);
        offset = rand() % divisor;

    The code only checks that n is positive.  If n is 1, the divisor becomes
    zero, and the CPU raises a divide-by-zero exception at IDIV.

The patch:

    Keep the existing rand() call exactly where it is, then replace the fragile
    x87 floating-point "multiply by 0.5 and convert to int" sequence with:

        divisor = n >> 1;        # same as n / 2 for the positive n values here
        if divisor == 0:
            remainder = 0;
        else:
            remainder = rand_result % divisor;

    This is intentionally more conservative than changing the earlier branch to
    match Enhanced Edition's "n > 3" guard.  For all non-crashing Diamond inputs
    it preserves the old divisor and the old rand() consumption.  It only changes
    the n == 1 case from "crash" to "offset 0".

Supported binaries:

    * Windows NWN Diamond 1.69-ish nwmain.exe builds with the vulnerable blocks
      observed at VAs 0x007DFE74, 0x007DFEA7, 0x007E972E, and 0x007E9759.

    * Linux x86 NWN Diamond client builds matching the function shown in the
      crash analysis, with the vulnerable blocks at VAs 0x08534033 and
      0x0853406D.

The patcher is signature-checked.  It refuses to write unless the original bytes
match exactly, or unless a site is already patched.

Usage:

    # Dry run; prints what would be changed.
    python patch_diamond_particle_sigfpe.py "C:\\NeverwinterNights\\NWN\\nwmain.exe"

    # Patch in place; creates "nwmain.exe.bak" first.
    python patch_diamond_particle_sigfpe.py "C:\\NeverwinterNights\\NWN\\nwmain.exe" --apply

    # Write a patched copy instead of modifying the input file.
    python patch_diamond_particle_sigfpe.py ./nwmain --output ./nwmain.patched
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import struct
import sys
from pathlib import Path


NOP = b"\x90"


@dataclasses.dataclass(frozen=True)
class PatchSite:
    """One exact byte replacement at a virtual address."""

    name: str
    va: int
    # Most sites have one expected byte sequence.  The Linux client reported by
    # a user uses an equivalent alternate encoding for `mov ecx, edx`, so a site
    # may also carry multiple acceptable original byte sequences.
    original: bytes | tuple[bytes, ...]
    replacement: bytes

    @property
    def originals(self) -> tuple[bytes, ...]:
        """Return all accepted original byte sequences for this site."""
        if isinstance(self.original, bytes):
            variants = (self.original,)
        else:
            variants = self.original

        lengths = {len(variant) for variant in variants}
        if len(lengths) != 1:
            raise ValueError(f"{self.name}: original variants must have the same length")
        return variants

    @property
    def original_length(self) -> int:
        """Length of the byte range replaced at this site."""
        return len(self.originals[0])

    @property
    def padded_replacement(self) -> bytes:
        """Return replacement padded with NOPs to exactly cover original."""
        if len(self.replacement) > self.original_length:
            raise ValueError(f"{self.name}: replacement is longer than original")
        return self.replacement + (NOP * (self.original_length - len(self.replacement)))


def hx(data: bytes) -> str:
    """Pretty hex for diagnostics."""
    return data.hex(" ")


def mov_ecx_from_esp(offset: int) -> bytes:
    """Encode `mov ecx, [esp+offset]` for the small stack offsets we use."""
    if not 0 <= offset <= 0x7F:
        raise ValueError("This helper only supports short [esp+imm8] forms")
    return bytes((0x8B, 0x4C, 0x24, offset))


def safe_divisor_patch(load_ecx: bytes) -> bytes:
    """
    Build the tiny replacement block.

    Input assumptions at each patch site:

        * EAX already contains the result of the existing rand() call.
        * The original integer spread, n, is still available in a register or
          stack slot.
        * Control only reaches this block after the original "n > 0" guard.

    Output contract:

        * EDX contains the random remainder, or zero if n / 2 is zero.
        * EAX/ECX may be clobbered, as in the original code.

    Machine code:

        load_ecx                 ; ecx = n
        sar ecx, 1               ; ecx = n / 2, ZF=1 if divisor is zero
        jnz do_divide
        xor edx, edx             ; zero-divisor case: offset = 0
        jmp done
      do_divide:
        cdq
        idiv ecx
      done:

    Note the ordering: we must branch on the flags from SAR.  An earlier version
    of this idea that did XOR EDX, EDX before the branch would clobber ZF and
    would skip every divide.
    """
    return b"".join(
        [
            load_ecx,
            b"\xD1\xF9",  # sar ecx, 1
            b"\x75\x04",  # jnz +4, to cdq
            b"\x33\xD2",  # xor edx, edx
            b"\xEB\x03",  # jmp +3, past cdq/idiv
            b"\x99",  # cdq
            b"\xF7\xF9",  # idiv ecx
        ]
    )


def linux_x87_half_divisor_original(push: int, pop: int, mov_ecx_edx: bytes) -> bytes:
    """
    Build the original Linux x87 half-divisor sequence.

    IDA displays both `8B CA` and `89 D1` as `mov ecx, edx`.  They are equivalent
    encodings of the same operation:

        8B CA    mov ecx, edx   ; MOV r32, r/m32
        89 D1    mov ecx, edx   ; MOV r/m32, r32

    The public Linux client reported by a user contains `89 D1`, while the first
    signature in this patcher used `8B CA`.  Accepting both keeps the safety of
    exact signatures without rejecting a semantically identical build.
    """
    return b"".join(
        [
            bytes((push,)),  # push esi/ebx for the original FIMUL [esp]
            bytes.fromhex("D9 05 84 5F 5F 08"),  # fld ds:flt_85F5F84 ; 0.5
            bytes.fromhex("DA 0C 24"),  # fimul dword ptr [esp]
            bytes.fromhex("D9 7D E4"),  # fnstcw [ebp-1Ch]
            bytes.fromhex("8B 4D E4"),  # mov ecx, [ebp-1Ch]
            bytes.fromhex("C6 45 E5 0C"),  # set x87 rounding mode to truncate
            bytes.fromhex("D9 6D E4"),  # fldcw [ebp-1Ch]
            bytes.fromhex("89 4D E4"),  # mov [ebp-1Ch], ecx
            bytes.fromhex("DB 5D E0"),  # fistp [ebp-20h]
            bytes.fromhex("D9 6D E4"),  # fldcw [ebp-1Ch]
            bytes.fromhex("8B 55 E0"),  # mov edx, [ebp-20h]
            mov_ecx_edx,
            bytes.fromhex("99"),  # cdq
            bytes((pop,)),  # pop esi/ebx
            bytes.fromhex("F7 F9"),  # idiv ecx
        ]
    )


WINDOWS_PATCHES = [
    PatchSite(
        name="Windows Particle::initialize x spread",
        va=0x007DFE74,
        original=bytes.fromhex(
            "DB 44 24 18"  # fild dword ptr [esp+18h]
            " 8B D8"  # mov ebx, eax
            " D8 0D 30 69 8A 00"  # fmul ds:flt_8A6930 ; 0.5
            " E8 67 C3 06 00"  # call float-to-int trunc helper
            " 8B C8"  # mov ecx, eax
            " 8B C3"  # mov eax, ebx
            " 99"  # cdq
            " F7 F9"  # idiv ecx
        ),
        replacement=safe_divisor_patch(mov_ecx_from_esp(0x18)),
    ),
    PatchSite(
        name="Windows Particle::initialize y spread",
        va=0x007DFEA7,
        original=bytes.fromhex(
            "DB 44 24 14"
            " 8B F8"
            " D8 0D 30 69 8A 00"
            " E8 34 C3 06 00"
            " 8B C8"
            " 8B C7"
            " 99"
            " F7 F9"
        ),
        replacement=safe_divisor_patch(mov_ecx_from_esp(0x14)),
    ),
    PatchSite(
        name="Windows emitter update x spread",
        va=0x007E972E,
        original=bytes.fromhex(
            "DB 44 24 14"
            " 8B E8"
            " D8 0D 30 69 8A 00"
            " E8 AD 2A 06 00"
            " 8B C8"
            " 8B C5"
            " 99"
            " F7 F9"
        ),
        replacement=safe_divisor_patch(mov_ecx_from_esp(0x14)),
    ),
    PatchSite(
        name="Windows emitter update y spread",
        va=0x007E9759,
        original=bytes.fromhex(
            "DB 44 24 30"
            " 8B F8"
            " D8 0D 30 69 8A 00"
            " E8 82 2A 06 00"
            " 8B C8"
            " 8B C7"
            " 99"
            " F7 F9"
        ),
        replacement=safe_divisor_patch(mov_ecx_from_esp(0x30)),
    ),
]


LINUX_PATCHES = [
    PatchSite(
        name="Linux Particle::randomPosition x spread",
        va=0x08534033,
        original=(
            linux_x87_half_divisor_original(0x56, 0x5E, bytes.fromhex("8B CA")),
            linux_x87_half_divisor_original(0x56, 0x5E, bytes.fromhex("89 D1")),
        ),
        replacement=safe_divisor_patch(b"\x8B\xCE"),  # mov ecx, esi
    ),
    PatchSite(
        name="Linux Particle::randomPosition y spread",
        va=0x0853406D,
        original=(
            linux_x87_half_divisor_original(0x53, 0x5B, bytes.fromhex("8B CA")),
            linux_x87_half_divisor_original(0x53, 0x5B, bytes.fromhex("89 D1")),
        ),
        replacement=safe_divisor_patch(b"\x8B\xCB"),  # mov ecx, ebx
    ),
]


class BinaryMapper:
    """Map virtual addresses to file offsets for a specific executable format."""

    def va_to_offset(self, va: int) -> int:
        raise NotImplementedError


class PeMapper(BinaryMapper):
    """Enough PE parsing to map Diamond nwmain.exe virtual addresses."""

    def __init__(self, data: bytes) -> None:
        if data[:2] != b"MZ":
            raise ValueError("not a PE file")

        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise ValueError("PE signature not found")

        self.image_base = struct.unpack_from("<I", data, pe_offset + 0x34)[0]
        section_count = struct.unpack_from("<H", data, pe_offset + 6)[0]
        optional_header_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
        section_table = pe_offset + 24 + optional_header_size

        self.sections: list[tuple[str, int, int, int, int]] = []
        for index in range(section_count):
            offset = section_table + index * 40
            name = data[offset : offset + 8].split(b"\0", 1)[0].decode("ascii", "replace")
            virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
                "<IIII", data, offset + 8
            )
            self.sections.append((name, virtual_address, virtual_size, raw_pointer, raw_size))

    def va_to_offset(self, va: int) -> int:
        rva = va - self.image_base
        for name, section_rva, virtual_size, raw_pointer, raw_size in self.sections:
            size = max(virtual_size, raw_size)
            if section_rva <= rva < section_rva + size:
                return raw_pointer + (rva - section_rva)
        raise ValueError(f"VA 0x{va:08X} is not inside a PE section")


class Elf32Mapper(BinaryMapper):
    """Enough ELF32 little-endian parsing to map the Linux Diamond client."""

    def __init__(self, data: bytes) -> None:
        if data[:4] != b"\x7FELF":
            raise ValueError("not an ELF file")
        if data[4] != 1:
            raise ValueError("only ELF32 is supported")
        if data[5] != 1:
            raise ValueError("only little-endian ELF is supported")

        program_header_offset = struct.unpack_from("<I", data, 0x1C)[0]
        program_header_entry_size = struct.unpack_from("<H", data, 0x2A)[0]
        program_header_count = struct.unpack_from("<H", data, 0x2C)[0]

        self.loads: list[tuple[int, int, int, int]] = []
        for index in range(program_header_count):
            offset = program_header_offset + index * program_header_entry_size
            (
                p_type,
                p_offset,
                p_vaddr,
                _p_paddr,
                p_filesz,
                p_memsz,
                _p_flags,
                _p_align,
            ) = struct.unpack_from("<IIIIIIII", data, offset)

            PT_LOAD = 1
            if p_type == PT_LOAD:
                self.loads.append((p_vaddr, p_memsz, p_offset, p_filesz))

    def va_to_offset(self, va: int) -> int:
        for vaddr, mem_size, file_offset, file_size in self.loads:
            if vaddr <= va < vaddr + mem_size:
                delta = va - vaddr
                if delta >= file_size:
                    raise ValueError(f"VA 0x{va:08X} maps past file-backed ELF data")
                return file_offset + delta
        raise ValueError(f"VA 0x{va:08X} is not inside an ELF PT_LOAD segment")


def detect_format(data: bytes) -> tuple[str, BinaryMapper, list[PatchSite]]:
    """Detect the executable type and return the matching mapper and patch set."""
    if data[:2] == b"MZ":
        return "Windows PE", PeMapper(data), WINDOWS_PATCHES
    if data[:4] == b"\x7FELF":
        return "Linux ELF32", Elf32Mapper(data), LINUX_PATCHES
    raise ValueError("input is neither a PE executable nor an ELF executable")


def check_and_patch(data: bytearray, mapper: BinaryMapper, patches: list[PatchSite]) -> tuple[int, int]:
    """
    Validate all sites first, then patch.

    The two-pass structure matters: if one site does not match, abort before
    changing any other site, leaving the executable untouched.
    """
    planned: list[tuple[PatchSite, int, bytes]] = []
    already_patched = 0

    for site in patches:
        offset = mapper.va_to_offset(site.va)
        current = bytes(data[offset : offset + site.original_length])
        patched = site.padded_replacement

        if current == patched:
            already_patched += 1
            print(f"already patched: {site.name} at VA 0x{site.va:08X}")
            continue

        if current not in site.originals:
            expected = "\n            ".join(hx(original) for original in site.originals)
            raise ValueError(
                "\n".join(
                    [
                        f"byte signature mismatch at {site.name}",
                        f"  VA:       0x{site.va:08X}",
                        f"  offset:   0x{offset:X}",
                        f"  expected: {expected}",
                        f"  found:    {hx(current)}",
                        "Refusing to patch this binary.",
                    ]
                )
            )

        planned.append((site, offset, patched))

    for site, offset, patched in planned:
        data[offset : offset + len(patched)] = patched
        print(f"patched: {site.name} at VA 0x{site.va:08X} / file offset 0x{offset:X}")

    return len(planned), already_patched


def copy_backup(path: Path) -> Path:
    """Create a simple .bak backup next to the input binary."""
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    shutil.copy2(path, backup)
    return backup


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Patch NWN Diamond particle emitter divide-by-zero sites."
    )
    parser.add_argument("binary", type=Path, help="Path to nwmain.exe or the Linux nwmain binary")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Modify the input file in place. Without this, the tool is a dry run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a patched copy here instead of modifying the input file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="When using --apply, do not create a .bak file first.",
    )
    args = parser.parse_args(argv)

    if args.apply and args.output:
        parser.error("--apply and --output are mutually exclusive")

    input_path = args.binary
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    original_data = input_path.read_bytes()
    format_name, mapper, patches = detect_format(original_data)
    print(f"detected: {format_name}")
    print(f"input:    {input_path}")
    print(f"sites:    {len(patches)}")

    patched_data = bytearray(original_data)
    changed, already = check_and_patch(patched_data, mapper, patches)

    if changed == 0:
        print(f"nothing to write: {already} site(s) already patched")
        return 0

    if args.output:
        args.output.write_bytes(patched_data)
        print(f"wrote patched copy: {args.output}")
        return 0

    if not args.apply:
        print("dry run only: pass --apply to patch in place, or --output to write a patched copy")
        return 0

    if not args.no_backup:
        backup = copy_backup(input_path)
        print(f"backup:   {backup}")

    input_path.write_bytes(patched_data)
    print(f"updated:  {input_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
