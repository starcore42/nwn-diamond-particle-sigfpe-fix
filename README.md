# NWN Diamond Particle SIGFPE Fix

Small binary patcher for a Neverwinter Nights Diamond client crash in the particle/VFX system.

It fixes a divide-by-zero bug where the original client can compute `rand() % 0` while placing particles for very small emitter sizes. This has been observed as a `SIGFPE` crash on Linux and the same vulnerable code pattern is present in the Windows Diamond binary.

Thanks to **kralex** for discovering the cause and location of the original bug.

## Usage

Dry run:

```sh
python patch_diamond_particle_sigfpe.py /path/to/nwmain
```

Patch in place, creating a `.bak` backup first:

```sh
python patch_diamond_particle_sigfpe.py /path/to/nwmain --apply
```

Write a patched copy instead:

```sh
python patch_diamond_particle_sigfpe.py /path/to/nwmain --output /path/to/nwmain.patched
```

The patcher checks exact byte signatures before writing and refuses to patch unknown binaries.
