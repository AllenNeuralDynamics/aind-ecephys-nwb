# Export subject to NWB
## NWB_Packaging_Subject_Capsule


### Description

This capsule is designed to append device, electrodes, and electrical series (raw and LFP) information to an existing NWB file.


### Inputs

The `data/` folder must include the session dataset and optionally the following JSON files:

- `subject.json`: Subject information following the [aind-data-schema]() specification
- `data_description.json`: session information following the [aind-data-schema]() specification

The `data_description.json` is only used to access the `name` field, which is used as session name.
An minimal example `data_description.json` file:

```json
{
    "name": "my-awesome-session-name"
}
```

If these files are missing, a mock NWB file is created.

If the `data/` folder includes an NWB file, this is copied to the results folder.


### Parameters

The `code/run` script takes 2 arguments:

```bash
  --backend {hdf5,zarr}
                        NWB backend. It can be either 'hdf5' or 'zarr'.
  --asset-name ASSET_NAME
                        Path to the data asset of the session. When provided, the metadata are fetched from the AIND metadata database. If None, and the attached data asset is used to fetch relevant metadata.
```

The `--asset-name` input is only used at AIND to fetch metadata from the database.


### Output

The output of this capsule is a NWB file in the `results/` folder named:
`{session_name}.nwb`

In case the `data/` folder contains a single NWB file, this is copied to the `results` with the same name.