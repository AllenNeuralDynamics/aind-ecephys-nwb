""" Writes RAW ephys and LFP to an NWB file """
import argparse
from pathlib import Path
import numpy as np
import json

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import probeinterface as pi

from neo.rawio import OpenEphysBinaryRawIO

from neuroconv.tools.nwb_helpers import (
    configure_backend,
    get_default_backend_configuration,
)
from neuroconv.tools.spikeinterface import add_recording

from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Device, Subject
from hdmf_zarr import NWBZarrIO

# for NWB Zarr, let's use built-in compressors, so thay can be read without Python
from numcodecs import Blosc
from utils import get_devices_from_metadata

# hdf5 or zarr
STUB_TEST = False
STUB_SECONDS = 10
WRITE_LFP = True
WRITE_RAW = True
WRITE_NIDQ = False

DEBUG = True

# filter and resample LFP
lfp_filter_kwargs = dict(freq_min=0.1, freq_max=500)
lfp_sampling_rate = 2500

# default compressors
default_electrical_series_compressors = dict(hdf5="gzip", zarr=Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE))

# default event line from open ephys
data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

job_kwargs = dict(n_jobs=-1, progress_bar=True)
si.set_global_job_kwargs(**job_kwargs)


parser = argparse.ArgumentParser(description="Export Neuropixels data to NWB")
# positional arguments
stub_group = parser.add_mutually_exclusive_group()
stub_help = "Write a stub version for testing"
stub_group.add_argument("--stub", action="store_true", help=stub_help)
stub_group.add_argument("static_stub", nargs="?", default="false", help=stub_help)

stub_seconds_group = parser.add_mutually_exclusive_group()
stub_seconds_help = "Duration of stub recording"
stub_seconds_group.add_argument("--stub-seconds", default=10, help=stub_seconds_help)
stub_seconds_group.add_argument("static_stub_seconds", nargs="?", default="10", help=stub_help)

write_lfp_group = parser.add_mutually_exclusive_group()
write_lfp_help = "Whether to write LFP electrical series"
write_lfp_group.add_argument("--skip-lfp", action="store_true", help=write_lfp_help)
write_lfp_group.add_argument("static_write_lfp", nargs="?", default="true", help=write_lfp_help)

write_raw_group = parser.add_mutually_exclusive_group()
write_raw_help = "Whether to write RAW electrical series"
write_raw_group.add_argument("--skip-raw", action="store_true", help=write_raw_help)
write_raw_group.add_argument("static_write_raw", nargs="?", default="true", help=write_raw_help)

write_nidq_group = parser.add_mutually_exclusive_group()
write_nidq_help = "Whether to write NIDQ stream"
write_nidq_group.add_argument("--write-nidq", action="store_true", help=write_nidq_help)
write_nidq_group.add_argument("static_write_nidq", nargs="?", default="false", help=write_nidq_help)

lfp_temporal_subsampling_group = parser.add_mutually_exclusive_group()
lfp_temporal_subsampling_help = (
    "Ratio of input samples to output samples in time. Use 0 or 1 to keep all samples. Default is 2."
)
lfp_temporal_subsampling_group.add_argument("--lfp_temporal_factor", default=2, help=lfp_temporal_subsampling_help)
lfp_temporal_subsampling_group.add_argument("static_lfp_temporal_factor", nargs="?", help=lfp_temporal_subsampling_help)

lfp_spatial_subsampling_group = parser.add_mutually_exclusive_group()
lfp_spatial_subsampling_help = (
    "Controls number of channels to skip in spatial subsampling. Use 0 or 1 to keep all channels. Default is 4."
)
lfp_spatial_subsampling_group.add_argument("--lfp_spatial_factor", default=4, help=lfp_spatial_subsampling_help)
lfp_spatial_subsampling_group.add_argument("static_lfp_spatial_factor", nargs="?", help=lfp_spatial_subsampling_help)

lfp_highpass_filter_group = parser.add_mutually_exclusive_group()
lfp_highpass_filter_help = (
    "Cutoff frequency for highpass filter to apply to the LFP recorsings. Default is 0.1 Hz. Use 0 to skip filtering."
)
lfp_highpass_filter_group.add_argument("--lfp_highpass_freq_min", default=0.1, help=lfp_highpass_filter_help)
lfp_highpass_filter_group.add_argument("static_lfp_highpass_freq_min", nargs="?", help=lfp_highpass_filter_help)

# common median referencing for probes in agar
lfp_surface_channel_agar_group = parser.add_mutually_exclusive_group()
lfp_surface_channel_help = "Index of surface channel (e.g. index 0 corresponds to channel 1) of probe for common median referencing for probes in agar. Pass in as JSON string where key is probe and value is surface channel (e.g. \"{'ProbeA': 350, 'ProbeB': 360}\")"
lfp_surface_channel_agar_group.add_argument(
    "--surface_channel_agar_probes_indices", help=lfp_surface_channel_help, default="", type=str
)
lfp_surface_channel_agar_group.add_argument(
    "static_surface_channel_agar_probes_indices", help=lfp_surface_channel_help, nargs="?", type=str
)

if __name__ == "__main__":
    args = parser.parse_args()

    stub = args.stub or args.static_stub
    if args.stub:
        STUB_TEST = True
    else:
        STUB_TEST = True if args.static_stub == "true" else False
    STUB_SECONDS = float(args.stub_seconds) or float(args.static_stub)

    if args.skip_lfp:
        WRITE_LFP = False
    else:
        WRITE_LFP = True if args.static_write_lfp == "true" else False

    if args.skip_raw:
        WRITE_RAW = False
    else:
        WRITE_RAW = True if args.static_write_raw == "true" else False
    if args.write_nidq:
        WRITE_NIDQ = True
    else:
        WRITE_NIDQ = True if args.static_write_nidq == "true" else False

    TEMPORAL_SUBSAMPLING_FACTOR = args.static_lfp_temporal_factor or args.lfp_temporal_factor
    TEMPORAL_SUBSAMPLING_FACTOR = int(TEMPORAL_SUBSAMPLING_FACTOR)
    SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = args.static_lfp_spatial_factor or args.lfp_spatial_factor
    SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = int(SPATIAL_CHANNEL_SUBSAMPLING_FACTOR)
    HIGHPASS_FILTER_FREQ_MIN = args.static_lfp_highpass_freq_min or args.lfp_highpass_freq_min
    HIGHPASS_FILTER_FREQ_MIN = float(HIGHPASS_FILTER_FREQ_MIN)
    SURFACE_CHANNEL_AGAR_PROBES_INDICES = (
        args.static_surface_channel_agar_probes_indices or args.surface_channel_agar_probes_indices
    )
    if SURFACE_CHANNEL_AGAR_PROBES_INDICES != "":
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = json.loads(SURFACE_CHANNEL_AGAR_PROBES_INDICES)
    else:
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = None

    print(f"Running NWB conversion with the following parameters:")
    print(f"Stub test: {STUB_TEST}")
    print(f"Stub seconds: {STUB_SECONDS}")
    print(f"Write LFP: {WRITE_LFP}")
    print(f"Write RAW: {WRITE_RAW}")
    print(f"Write NIDQ: {WRITE_NIDQ}")
    print(f"Temporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
    print(f"Spatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
    print(f"Highpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
    print(f"Surface channel indices for agar probes: {SURFACE_CHANNEL_AGAR_PROBES_INDICES}")

    # find ecephys session
    sessions = [
        p.stem
        for p in data_folder.iterdir()
        if ("ecephys" in p.stem or "behavior" in p.stem) and "sorted" not in p.stem and "nwb" not in p.name
    ]
    assert len(sessions) == 1, "Attach one session (raw data) data at a time"
    session = sessions[0]
    ecephys_raw_folder = data_folder / session

    # find base NWB file
    nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"
    nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        NWB_SUFFIX = ".nwb.zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        NWB_SUFFIX = ".nwb"
        io_class = NWBHDF5IO
    print(f"NWB backend: {NWB_BACKEND}")

    print(f"\nExporting session: {session}")
    session_folder = data_folder / session

    if (data_folder / session / "ecephys_clipped").is_dir():
        oe_folder = data_folder / session / "ecephys_clipped"
        compressed_folder = data_folder / session / "ecephys_compressed"
    else:
        assert (data_folder / session / "ecephys").is_dir()
        if (data_folder / session / "ecephys" / "ecephys_compressed").is_dir():
            oe_folder = data_folder / session / "ecephys" / "ecephys_clipped"
            compressed_folder = data_folder / session / "ecephys" / "ecephys_compressed"
        else:
            oe_folder = data_folder / session / "ecephys"
            compressed_folder = None

    # Read Open Ephys Folder structure with NEO
    neo_io = OpenEphysBinaryRawIO(oe_folder)
    neo_io.parse_header()
    num_blocks = neo_io.block_count()
    print(f"Number of experiments: {num_blocks}")
    stream_names = neo_io.header["signal_streams"]["name"]
    record_nodes = list(neo_io.folder_structure.keys())
    experiment_ids = [eid for eid in neo_io.folder_structure[record_nodes[0]]["experiments"].keys()]
    experiment_names = [e["name"] for eid, e in neo_io.folder_structure[record_nodes[0]]["experiments"].items()]
    recording_names = [
        r["name"]
        for rid, r in neo_io.folder_structure[record_nodes[0]]["experiments"][experiment_ids[0]]["recordings"].items()
    ]

    streams_to_process = []
    for stream_name in stream_names:
        # Skip NI-DAQ if WRITE_NIDQ is False
        if "NI-DAQ" in stream_name and not WRITE_NIDQ:
            continue
        # LFP are handled later
        if "LFP" in stream_name:
            continue
        streams_to_process.append(stream_name)
    print(f"Number of streams to write: {len(streams_to_process)}")

    if DEBUG:
        streams_to_process = streams_to_process[:2]

    # Construct 1 nwb file per experiment - streams are concatenated!
    nwb_output_files = []
    electrical_series_to_configure = []
    for block_index in range(num_blocks):
        experiment_name = experiment_names[block_index]

        num_segments = neo_io.segment_count(block_index)
        for segment_index in range(num_segments):
            recording_name = recording_names[segment_index]

            nwbfile_out_name = f"{session}_{experiment_name}_experiment_name"

            if STUB_TEST:
                nwbfile_out_name = f"{nwbfile_out_name}_stub"

            nwbfile_output_path = results_folder / f"{nwbfile_out_name}{NWB_SUFFIX}"

            # write 1 new nwb file per segment
            with io_class(str(nwbfile_input_path), "r") as read_io:
                nwbfile = read_io.read()

                for stream_name in streams_to_process:
                    record_node, oe_stream_name = stream_name.split("#")
                    recording_folder_name = f"{experiment_name}_{stream_name}_{recording_name}"
                    settings_file = neo_io.folder_structure[record_node]["experiments"][experiment_ids[block_index]][
                        "settings_file"
                    ]

                    # Add devices
                    added_devices, target_locations = get_devices_from_metadata(
                        session_folder, segment_index=segment_index
                    )

                    # if devices not found in metadata, instantiate using probeinterface
                    if added_devices:
                        for device_name, device in added_devices.items():
                            if device_name not in nwbfile.devices:
                                nwbfile.add_device(device)
                        for device_name, targeted_location in target_locations.items():
                            probe_no_spaces = device_name.replace(" ", "")
                            if probe_no_spaces in oe_stream_name:
                                probe_device_name = probe_no_spaces
                                electrode_group_location = targeted_location
                                print(f"Found device from rig: {probe_device_name}")
                                break
                    else:
                        electrode_group_location = "unknown"
                        probe = pi.read_openephys(settings_file, stream_name=oe_stream_name)
                        probe_device_name = probe.name
                        probe_device_description = f"Model: {probe.model_name} - Serial number: {probe.serial_number}"
                        probe_device_manufacturer = f"{probe.manufacturer}"
                        probe_device = Device(
                            name=probe_device_name,
                            description=probe_device_description,
                            manufacturer=probe_device_manufacturer,
                        )
                        if probe_device_name not in nwbfile.devices:
                            nwbfile.add_device(probe_device)
                            print(f"\tAdded probe device: {probe_device.name} from probeinterface")

                    # Add Recordings
                    if compressed_folder is not None:
                        stream_name_zarr = f"{experiment_name}_{stream_name}"
                        recording_multi_segment = si.read_zarr(compressed_folder / f"{stream_name_zarr}.zarr")
                    else:
                        recording_multi_segment = se.read_openephys(
                            oe_folder, stream_name=stream_name, block_index=block_index
                        )

                    recording = si.split_recording(recording_multi_segment)[segment_index]

                    electrode_metadata = dict(
                        Ecephys=dict(
                            Device=[dict(name=probe_device_name)],
                            ElectrodeGroup=[
                                dict(
                                    name=probe_device_name,
                                    description=f"Recorded electrodes from probe {probe_device_name}",
                                    location=electrode_group_location,
                                    device=probe_device_name,
                                )
                            ],
                        )
                    )

                    if WRITE_RAW:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}",
                                description=f"Voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}", write_as="raw"
                        )
                        # Add channel properties (group_name property to associate electrodes with group)
                        recording.set_channel_groups([probe_device_name] * recording.get_num_channels())

                        if STUB_TEST:
                            end_frame = int(STUB_SECONDS * recording.sampling_frequency)
                            recording = recording.frame_slice(start_frame=0, end_frame=end_frame)

                        print(f"\tAdding RAW data for stream {stream_name} - segment {segment_index}")
                        add_recording(
                            recording=recording,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            compression=None,
                            **add_electrical_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_series_kwargs["es_key"])

                    if WRITE_LFP:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}-LFP"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}-LFP",
                                description=f"LFP voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_lfp_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}-LFP",
                            write_as="lfp",
                        )

                        if "AP" not in stream_name:
                            # Wide-band NP recording: filter and resample LFP
                            print(
                                f"\tAdding LFP data for stream {stream_name} from wide-band signal - segment {segment_index}"
                            )
                            recording_lfp = spre.bandpass_filter(recording, **lfp_filter_kwargs)
                            recording_lfp = spre.resample(recording_lfp, lfp_sampling_rate)
                            recording_lfp = spre.scale(recording_lfp, dtype="int16")

                            # there is a bug in with sample mismatches for the last chunk if num_samples not divisible by chunk_size
                            # the workaround is to discard the last samples to make it "even"
                            if recording.get_num_segments() == 1:
                                recording_lfp = recording_lfp.frame_slice(
                                    start_frame=0,
                                    end_frame=int(
                                        recording_lfp.get_num_samples() // lfp_sampling_rate * lfp_sampling_rate
                                    ),
                                )
                            lfp_period = 1.0 / lfp_sampling_rate
                            for segment_index in range(recording.get_num_segments()):
                                ts_lfp = (
                                    np.arange(recording_lfp.get_num_samples(segment_index))
                                    / recording_lfp.sampling_frequency
                                    - recording.get_times(segment_index)[0]
                                    + lfp_period / 2
                                )
                                recording_lfp.set_times(ts_lfp, segment_index=segment_index, with_warning=False)
                        else:
                            # load LFP recording for NP1.0 probes
                            lfp_stream_name = stream_name.replace("AP", "LFP")
                            print(f"\tAdding LFP data for {lfp_stream_name} - segment {segment_index}")
                            if compressed_folder is not None:
                                stream_name_zarr = f"{experiment_name}_{lfp_stream_name}"
                                recording_lfp = si.read_zarr(
                                    ecephys_raw_folder / "ecephys_compressed" / f"{stream_name_zarr}.zarr"
                                )
                            else:
                                recording_lfp = se.read_openephys(
                                    oe_folder, stream_name=lfp_stream_name, block_index=block_index
                                )

                        channel_ids = recording_lfp.get_channel_ids()

                        # re-reference only for agar - subtract median of channels out of brain using surface channel index arg
                        # similar processing to allensdk
                        if SURFACE_CHANNEL_AGAR_PROBES_INDICES is not None:
                            if probe.name in SURFACE_CHANNEL_AGAR_PROBES_INDICES:
                                surface_channel_index = SURFACE_CHANNEL_AGAR_PROBES_INDICES[probe.name]
                                # get indices of channels out of brain including surface channel
                                reference_channel_indices = np.arange(surface_channel_index, len(channel_ids), 1)
                                reference_channel_ids = channel_ids[reference_channel_indices]
                                groups = [[channel_id] for channel_id in reference_channel_ids]
                                # common median reference to channels out of brain
                                recording_lfp = spre.common_reference(
                                    recording_lfp,
                                    reference="single",
                                    groups=groups,
                                    ref_channel_ids=reference_channel_ids,
                                )

                        # spatial subsampling from allensdk - keep every nth channel
                        if SPATIAL_CHANNEL_SUBSAMPLING_FACTOR > 1:
                            channel_ids_to_keep = channel_ids[0 : len(channel_ids) : SPATIAL_CHANNEL_SUBSAMPLING_FACTOR]
                            recording_lfp = recording_lfp.channel_slice(channel_ids_to_keep)

                        # time subsampling/decimate
                        if TEMPORAL_SUBSAMPLING_FACTOR > 1:
                            recording_lfp = spre.decimate(recording_lfp, TEMPORAL_SUBSAMPLING_FACTOR)

                        # high pass filter from allensdk
                        if HIGHPASS_FILTER_FREQ_MIN > 0:
                            recording_lfp = spre.highpass_filter(recording_lfp, freq_min=HIGHPASS_FILTER_FREQ_MIN)

                        # Assign to the correct channel group
                        recording_lfp.set_channel_groups([probe_device_name] * recording_lfp.get_num_channels())

                        if STUB_TEST:
                            end_frame = int(STUB_SECONDS * recording_lfp.sampling_frequency)
                            recording_lfp = recording_lfp.frame_slice(start_frame=0, end_frame=end_frame)

                        # For NP2, save the LFP to speed up conversion later
                        if "AP" not in stream_name:
                            recording_lfp = recording_lfp.save(folder=scratch_folder / f"{recording_folder_name}-LFP")

                        add_recording(
                            recording=recording_lfp,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            compression=None,
                            **add_electrical_lfp_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_lfp_series_kwargs["es_key"])

                print(f"Added {len(streams_to_process)} streams")

                print(f"Configuring {NWB_BACKEND} backend")
                backend_configuration = get_default_backend_configuration(nwbfile=nwbfile, backend=NWB_BACKEND)
                es_compressor = default_electrical_series_compressors[NWB_BACKEND]

                for key in backend_configuration.dataset_configurations.keys():
                    if any([es_name in key for es_name in electrical_series_to_configure]) and "timestamps" not in key:
                        backend_configuration.dataset_configurations[key].compression_method = es_compressor
                        print(f"\tSetting compression for {key} to {es_compressor}")
                configure_backend(nwbfile=nwbfile, backend_configuration=backend_configuration)

                with io_class(str(nwbfile_output_path), "w") as export_io:
                    export_io.export(src_io=read_io, nwbfile=nwbfile)
                print(f"Done writing {nwbfile_output_path}")
                nwb_output_files.append(nwbfile_output_path)
