# Raw Data

This directory is reserved for unmodified raw data files for the project.

## Purpose

- **Preservation**: Store original, unmodified data files
- **Backup**: Maintain copies of data before any processing
- **Reproducibility**: Ensure original data remains unchanged

## Usage Guidelines

- **Do not edit** files in this folder
- **Do not process** data directly from this folder  
- Use `../external` for data ready for analysis
- See `../processed` for analysis outputs

## Data Flow

```
raw/ → external/ → processed/
 ↓         ↓           ↓
backup   input     outputs
```

**Workflow:**
1. `raw/`: Original unmodified data (backup)
2. `external/`: Input data ready for analysis
3. `processed/`: Analysis results and outputs

## Current Status

This directory is currently empty as the project uses data directly from the `external/` directory. Raw data backups can be stored here if needed for data provenance and reproducibility.

## Best Practices

- Always keep original data files unchanged
- Document any data modifications in processing scripts
- Maintain clear data lineage from raw → external → processed