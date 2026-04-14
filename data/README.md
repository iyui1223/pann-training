# Training Data

Place `training_dataset_partitioned.nc` in this directory.

Download from:
- **Google Drive**: *TODO — paste link*
- **Hugging Face**: *TODO — paste link*

## File specification

- Format: NetCDF-4
- Dimensions: `sample` (~9,874) x `level` (91)
- Size: ~200 MB

### Key variables

| Category | Variables |
|----------|-----------|
| Convection tendencies | `{T,Q,QL,QI}_tendency_conv` |
| Cloud tendencies | `{T,Q,QL,QI}_tendency_cloud` |
| Radiation heating rates | `PHRSW`, `PHRSC`, `PHRLW`, `PHRLC` |
| Vertical diffusion | `{T,Q,QL,QI}_tendency_vdif` |
| Non-orographic GWD | `{T,Q,QL,QI}_tendency_nogw` |
| Total physics tendency | `{T,Q}_tendency_total` |
| Actual tendency | `{T,Q}_tendency_actual` |
| Initial state | `{T,Q,QL,QI}_initial`, `pressure_initial` |

### Provenance

- **SCM**: OpenIFS 48r1 (single-column mode)
- **Forcing**: Christensen et al. (2018) CEDA archive (~20S-20N, ~42E-177E)
- **Timestep**: 900 s; output after 4 physics steps (1 hour)
- **Levels**: L91 (IFS standard)
