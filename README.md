# Export ecephys to NWB
## NWB Packaging Ecephys


### Description

This capsule is designed to append device, electrodes, LFP and raw (optional) electrical series information to an existing NWB file.


### Inputs

The `data/` folder must include:

- the raw data in any supported format (e.g. AIND ephys format, OpenEphys, SpikeGLX, etc.)
- (optional) a list of json files produced by the [aind-ephys-job-dispatch](https://github.com/AllenNeuralDynamics/aind-ephys-job-dispatch/) process

If the input data is already in NWB format, the capsule will simply start from the existing NWB file and append the ecephys data. If in another format, the capsule will look for the ``subject.json`` and 
``data_description.json`` files in the `data/` folder to extract subject and session metadata. If these files are not found, the capsule will create mock subject and session metadata.

### Parameters

Parameters are read from a `params.json` file that must be in the root folder of the capsule (next to `code/`, i.e. `../params.json` from `code/`): the `nwb.ecephys` section plus the top-level `logging` section.
Keys: `backend` (`zarr`|`hdf5`, default `zarr`), `stub` (false), `stub_seconds` (10), `write_lfp` (true), `write_raw` (false),
`lfp_temporal_factor` (2), `lfp_spatial_factor` (4), `lfp_highpass_freq_min` (0.1; 0 to skip),
`surface_channel_agar_probes_indices` (dict of probe name to surface channel index, for common median referencing of probes in agar).

### Output

The output of this capsule is a set of NWB files in the `results/` folder named associated with different experiments/blocks and segments.
