# Export ecephys to NWB
## NWB Packaging Ecephys


### Description

This capsule is designed to append device, electrodes, LFP and raw (optional) electrical series information to an existing NWB file.


### Inputs

The `data/` folder must include:

- a base valid NWB file to append the ephys data to
- the raw data
- (optional) a list of json files produced by the [aind-ephys-job-dispatch](https://github.com/AllenNeuralDynamics/aind-ephys-job-dispatch/) process

If no JSON files are provided, the capsule will assume the raw data is in the [AIND
ephys format](https://github.com/AllenNeuralDynamics/aind-physio-arch/blob/main/doc/file_formats/ephys.md). When JSON files are provided, the capsule will use this information 
to load the raw and LFP data.

### Parameters

The `code/run` script takes the following arguments:

```bash
  --stub                Write a stub version for testing
  --stub-seconds STUB_SECONDS
                        Duration of stub recording
  --skip-lfp            Whether to write LFP electrical series
  --write-raw           Whether to write RAW electrical series
  --lfp_temporal_factor LFP_TEMPORAL_FACTOR
                        Ratio of input samples to output samples in time. Use 0 or 1 to keep all samples. Default is 2.
  --lfp_spatial_factor LFP_SPATIAL_FACTOR
                        Controls number of channels to skip in spatial subsampling. Use 0 or 1 to keep all channels. Default is 4.
  --lfp_highpass_freq_min LFP_HIGHPASS_FREQ_MIN
                        Cutoff frequency for highpass filter to apply to the LFP recorsings. Default is 0.1 Hz. Use 0 to skip filtering.
  --surface_channel_agar_probes_indices SURFACE_CHANNEL_AGAR_PROBES_INDICES
                        Index of surface channel (e.g. index 0 corresponds to channel 1) of probe for common median referencing for probes in agar. Pass in as JSON
                        string where key is probe and value is surface channel (e.g. "{'ProbeA': 350, 'ProbeB': 360}")
```

### Output

The output of this capsule is a set of NWB files in the `results/` folder named associated with different experiments/blocks and segments.
