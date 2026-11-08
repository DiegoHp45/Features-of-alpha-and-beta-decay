# Helmholtz EOS (Timmes & Swesty 2000)

This is the **official, complete** Helmholtz free-energy tabular EOS package from
F.X. Timmes (https://cococubed.com/code_pages/eos.shtml), fetched 2026-08-06.

## Files
- `helmholtz.f90` — main program (`teos`) + `read_helm_table` + `helmeos` (the
  full EOS: electron-positron table lookup + analytic ion gas + analytic
  radiation, combined thermodynamically consistently) + pretty-printer.
- `helm_table.dat` — the precomputed Helmholtz free-energy table (541×201 grid
  in log10(den) × log10(temp), ~60 MB, plain text). Required at run time;
  loaded once by `read_helm_table`.
- `const.dek`, `implno.dek`, `vector_eos.dek`, `helm_table_storage.dek` —
  `include`d parameter/common-block files (physical constants, `implicit
  none`, the pipeline arrays, and the table storage arrays respectively).
- `README_helmholtz.txt` — the original one-line author README.

## Build & run
```bash
cd HelmholtzEOS
gfortran -O2 -ffree-line-length-none helmholtz.f90 -o helmholtz.exe
./helmholtz.exe
```
`helm_table.dat` must be in the working directory the executable is run from
(it's opened by a relative path in `read_helm_table`).

Verified working: compiles with zero warnings, runs in ~1.5s CPU (dominated by
the one-time 60 MB table read), and reports `maxw1/2/3` (the three Maxwell
thermodynamic-consistency residuals) at machine precision (~1e-16), matching
the paper's claim of perfect thermodynamic consistency.

## Calling it from other code
The subroutine `helmeos` operates in "pipeline" mode over the common-block
arrays declared in `vector_eos.dek`:
1. Fill `temp_row(1:n)`, `den_row(1:n)`, `abar_row(1:n)`, `zbar_row(1:n)` for
   up to `n <= nrowmax` (1000) points.
2. Set `jlo_eos = 1`, `jhi_eos = n`.
3. `call read_helm_table` once at startup.
4. `call helmeos` — results land in `ptot_row`, `etot_row`, `stot_row`,
   `dpt_row`, `det_row`, `dst_row`, `cv_row`, `cp_row`, `gam1_row`, `cs_row`,
   etc. (see `vector_eos.dek` for the full list).

`abar`/`zbar` are the mean nucleon number and mean charge of the material
(`ye = zbar/abar`). If your trajectory generator (`nsm_traj.py`) only tracks
`Y_e` rather than a full isotopic composition, you'll need to decide how to
map that to `(abar, zbar)` for this call (e.g. a single effective nucleus, or
carrying `abar` separately) — that mapping isn't implied by the paper, so it's
a project decision rather than something to guess here.

## Note on repo size
`helm_table.dat` is ~60 MB. That's fine on disk, but committing it to git will
permanently bloat the repository history. Consider `.gitignore`-ing it (and
documenting the download step instead), or using Git LFS, before committing —
ask if you'd like me to set that up.
