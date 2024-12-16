""" Writes RAW ephys and LFP to an NWB file """

import argparse
from pathlib import Path
import numpy as np
import os
import json
import time

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import probeinterface as pi

from neo.rawio import OpenEphysBinaryRawIO

from neuroconv.tools.nwb_helpers import (
    configure_backend,
    get_default_backend_configuration,
)
from neuroconv.tools.spikeinterface import (
    add_recording_to_nwbfile,
    add_electrodes_to_nwbfile,
)

from pynwb import NWBHDF5IO
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

# for NWB Zarr, let's use built-in compressors, so thay can be read without Python
from numcodecs import Blosc

# AIND
try:
    from aind_log_utils import log

    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

from utils import get_devices_from_rig_metadata


# filter and resample LFP
lfp_filter_kwargs = dict(freq_min=0.1, freq_max=500)
lfp_sampling_rate = 2500

# default compressors
default_electrical_series_compressors = dict(hdf5="gzip", zarr=Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE))

# default event line from open ephys
data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

n_jobs = os.cpu_count()
job_kwargs = dict(n_jobs=n_jobs, progress_bar=False)
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
write_raw_group.add_argument("--write-raw", action="store_true", help=write_raw_help)
write_raw_group.add_argument("static_write_raw", nargs="?", default="false", help=write_raw_help)

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
    print("\n\nNWB EXPORT ECEPHYS")
    t_export_start = time.perf_counter()

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

    if args.write_raw:
        WRITE_RAW = True
    else:
        WRITE_RAW = True if args.static_write_raw == "true" else False

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
    print(f"Temporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
    print(f"Spatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
    print(f"Highpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
    print(f"Surface channel indices for agar probes: {SURFACE_CHANNEL_AGAR_PROBES_INDICES}")

    # find base NWB file
    nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"
    nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
    print(f"NWB backend: {NWB_BACKEND}")

    # find raw data
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir()
        and ("ecephys" in p.name or "behavior" in p.name)
        and ("sorted" not in p.name and "nwb" not in p.name)
    ]
    assert len(ecephys_folders) == 1, "Attach one ecephys folder at a time"
    ecephys_session_folder = ecephys_folders[0]
    session_name = ecephys_session_folder.name
    if HAVE_AIND_LOG_UTILS:
        # look for subject.json and data_description.json files
        subject_json = ecephys_session_folder / "subject.json"
        subject_id = "undefined"
        if subject_json.is_file():
            subject_data = json.load(open(subject_json, "r"))
            subject_id = subject_data["subject_id"]

        data_description_json = ecephys_session_folder / "data_description.json"
        session_name = "undefined"
        if data_description_json.is_file():
            data_description = json.load(open(data_description_json, "r"))
            session_name = data_description["name"]

        log.setup_logging(
            "NWB Packaging Ecephys",
            mouse_id=subject_id,
            session_name=session_name,
        )

    print(f"\nExporting session: {session_name}")

    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")

    if len(job_dicts) == 0:
        print("Standalone mode!")
        # AIND-specific section to parse AIND files
        if (ecephys_session_folder / "ecephys_clipped").is_dir():
            oe_folder = ecephys_session_folder / "ecephys_clipped"
            compressed_folder = ecephys_session_folder / "ecephys_compressed"
        else:
            assert (ecephys_session_folder / "ecephys").is_dir()
            if (ecephys_session_folder / "ecephys" / "ecephys_compressed").is_dir():
                oe_folder = ecephys_session_folder / "ecephys" / "ecephys_clipped"
                compressed_folder = ecephys_session_folder / "ecephys" / "ecephys_compressed"
            else:
                oe_folder = ecephys_session_folder / "ecephys"
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
            for rid, r in neo_io.folder_structure[record_nodes[0]]["experiments"][experiment_ids[0]][
                "recordings"
            ].items()
        ]
        # in this case we don't need to split by group, since
        block_ids = experiment_names
        recording_ids = recording_names
    else:
        # we create a result NWB file for each experiment/recording
        recording_names = [job_dict["recording_name"] for job_dict in job_dicts]

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        stream_names = []
        for recording_name in recording_names:
            if "group" in recording_name:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-2]
                stream_name = "_".join(recording_name.split("_")[1:-2])
            else:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-1])

            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
        # note: in case of groups, we will need to aggregate the data for each stream into a single recording

    streams_to_process = []
    for stream_name in stream_names:
        # Skip NI-DAQ
        if "NI-DAQ" in stream_name:
            continue
        # LFP are handled later
        if "LFP" in stream_name:
            continue
        streams_to_process.append(stream_name)

    block_ids = sorted(block_ids)
    recording_ids = sorted(recording_ids)
    streams_to_process = sorted(streams_to_process)

    print(f"Number of NWB files to write: {len(block_ids) * len(recording_ids)}")

    print(f"Number of streams to write for each file: {len(streams_to_process)}")

    # Construct 1 nwb file per experiment - streams are concatenated!
    nwb_output_files = []
    electrical_series_to_configure = []
    nwb_output_files = []
    for block_index, block_str in enumerate(block_ids):
        for segment_index, recording_str in enumerate(recording_ids):
            # add recording/experiment id if needed
            nwb_original_file_name = nwbfile_input_path.stem
            if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                nwb_file_name = nwb_original_file_name
            else:
                nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}"

            if STUB_TEST:
                nwb_file_name = f"{nwb_file_name}_stub"

            nwbfile_output_path = results_folder / f"{nwb_file_name}.nwb"

            # Find probe devices (this will only work for AIND)
            devices_from_rig, target_locations = get_devices_from_rig_metadata(
                ecephys_session_folder, segment_index=segment_index
            )

            # write 1 new nwb file per segment
            with io_class(str(nwbfile_input_path), "r") as read_io:
                nwbfile = read_io.read()

                probe_device_names = []
                for stream_index, stream_name in enumerate(streams_to_process):
                    recording_name = f"{block_str}_{stream_name}_{recording_str}"
                    print(f"Processing {recording_name}")

                    # load JSON and recordings
                    # we need lists because multiple groups are saved to different JSON files
                    recording_job_dicts = []
                    for job_dict in job_dicts:
                        if recording_name in job_dict["recording_name"]:
                            recording_job_dicts.append(job_dict)

                    recording_lfp = None
                    if recording_job_dicts is not None:
                        recordings = []
                        recordings_lfp = []
                        print(f"\tLoading {recording_name} from {len(recording_job_dicts)} JSON files")
                        if len(recording_job_dicts) > 1:
                            # in case of multiple groups, sort by group names
                            sort_idxs = np.argsort([jd["recording_name"] for jd in recording_job_dicts])
                            recording_job_dicts_sorted = np.array(recording_job_dicts)[sort_idxs]
                        else:
                            recording_job_dicts_sorted = recording_job_dicts
                        for recording_job_dict in recording_job_dicts_sorted:
                            recording = si.load_extractor(recording_job_dict["recording_dict"], base_folder=data_folder)
                            skip_times = recording_job_dict.get("skip_times", False)
                            if skip_times:
                                recording.reset_times()
                            recordings.append(recording)
                            print(f"\t\t{recording_job_dict['recording_name']}: {recording}")
                            if "recording_lfp_dict" in job_dict:
                                print(f"\tLoading associated LFP recording")
                                recording_lfp = si.load_extractor(job_dict["recording_lfp_dict"], base_folder=data_folder)
                                if skip_times:
                                    recording_lfp.reset_times()
                                recordings_lfp.append(recording_lfp)
                                print(f"\t\t{recording_lfp}")

                        # for multiple groups, aggregate channels
                        if len(recording_job_dicts_sorted) > 1:
                            print(f"\t\tAggregating channels from {len(recordings)} groups")
                            recording = si.aggregate_channels(recordings)
                            # probes_info get lost in aggregation, so we need to manually set them
                            recording.annotate(
                                probes_info=recordings[0].get_annotation("probes_info")
                            )
                            if len(recordings_lfp) > 0:
                                recording_lfp = si.aggregate_channels(recordings_lfp)
                                recording_lfp.annotate(
                                    probes_info=recordings_lfp[0].get_annotation("probes_info")
                                )

                    else:
                        print("\tCould not find JSON file")
                        # Add Recordings
                        recording_multi_segment_lfp = None
                        if compressed_folder is not None:
                            stream_name_zarr = f"{block_str}_{stream_name}"
                            recording_multi_segment = si.read_zarr(compressed_folder / f"{stream_name_zarr}.zarr")
                            try:
                                stream_name_zarr_lfp = stream_name_zarr.replace("AP", "LFP")
                                recording_multi_segment_lfp = si.read_zarr(
                                    compressed_folder / f"{stream_name_zarr_lfp}.zarr"
                                )
                            except:
                                pass
                        else:
                            recording_multi_segment = se.read_openephys(
                                oe_folder, stream_name=stream_name, block_index=block_index
                            )
                            try:
                                recording_multi_segment_lfp = se.read_openephys(
                                    oe_folder, stream_name=stream_name.replace("AP", "LFP"), block_index=block_index
                                )
                            except:
                                pass
                        print(f"\tLoading recording from AIND raw data")
                        recording = si.split_recording(recording_multi_segment)[segment_index]
                        print(f"\t\t{recording}")
                        if recording_multi_segment_lfp is not None:
                            print(f"\tLoading associated LFP recording")
                            recording_lfp = si.split_recording(recording_multi_segment_lfp)[segment_index]
                            print(f"\t\t{recording_lfp}")
                        else:
                            recording_lfp = None

                    # Add device and electrode group
                    probe_device_name = None
                    if devices_from_rig:
                        for device_name, device in devices_from_rig.items():
                            # add the device, since it could be a laser
                            if device_name not in nwbfile.devices:
                                nwbfile.add_device(device)
                            # find probe device name
                            probe_no_spaces = device_name.replace(" ", "")
                            if probe_no_spaces in stream_name:
                                probe_device_name = device_name
                                electrode_group_location = target_locations.get(device_name, "unknown")
                                print(
                                    f"Found device from rig: {probe_device_name} at location {electrode_group_location}"
                                )
                                break
                    probe_info = None
                    if probe_device_name is None:
                        if recording_job_dict is not None:
                            electrode_group_location = "unknown"
                            probes_info = recording.get_annotation("probes_info", None)
                            if probes_info is not None and len(probes_info) == 1:
                                probe_info = probes_info[0]
                        else:
                            electrode_group_location = "unknown"
                            record_node, oe_stream_name = stream_name.split("#")
                            recording_folder_name = f"{block_str}_{stream_name}_{recording_name}"
                            settings_file = neo_io.folder_structure[record_node]["experiments"][
                                experiment_ids[block_index]
                            ]["settings_file"]
                            probe = pi.read_openephys(settings_file, stream_name=oe_stream_name)
                            probe_info = dict(
                                name=probe.name,
                                manuefacturer=probe.manufacturer,
                                model_name=probe.model_name,
                                serial_number=probe.serial_number,
                            )

                    if probe_info is not None:
                        probe_device_name = probe_info.get("name", None)
                        probe_device_manufacturer = probe_info.get("manufacturer", None)
                        probe_model_name = probe_info.get("model_name", None)
                        probe_serial_number = probe_info.get("serial_number", None)
                        probe_device_description = ""
                        if probe_device_name is None:
                            if probe_model_name is not None:
                                probe_device_name = probe_model_name
                            else:
                                probe_device_name = "Probe"
                        if probe_model_name is not None:
                            probe_device_description += f"Model: {probe_device_description}"
                        if probe_serial_number is not None:
                            if len(probe_device_description) > 0:
                                probe_device_description += " - "
                            probe_device_description += f"Serial number: {probe_serial_number}"
                        # this is needed to account for a case where multiple streams have the same device name
                        if len(streams_to_process) > 1 and probe_device_name in probe_device_names:
                            probe_device_name = f"{probe_device_name}-{stream_index}"
                        probe_device = Device(
                            name=probe_device_name,
                            description=probe_device_description,
                            manufacturer=probe_device_manufacturer,
                        )
                        if probe_device_name not in nwbfile.devices:
                            nwbfile.add_device(probe_device)
                            print(f"\tAdded probe device: {probe_device.name} from recording metadata")
                    # last resort: could not find a device
                    if probe_device_name is None:
                        print("\tCould not load device information: using default Device")
                        probe_device_name = "Device"
                        if len(streams_to_process) > 1 and probe_device_name in probe_device_names:
                            probe_device_name = f"{probe_device_name}-{stream_index}"
                        probe_device = Device(name=probe_device_name, description="Default device")

                    # keep track of all added probe device names
                    probe_device_names.append(probe_device_name)

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
                    # Add channel properties (group_name property to associate electrodes with group)
                    recording.set_channel_groups([probe_device_name] * recording.get_num_channels())
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

                        if STUB_TEST:
                            end_frame = int(STUB_SECONDS * recording.sampling_frequency)
                            recording = recording.frame_slice(start_frame=0, end_frame=end_frame)

                        print(f"\tAdding RAW data for stream {stream_name} - segment {segment_index}")
                        add_recording_to_nwbfile(
                            recording=recording,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            compression=None,
                            always_write_timestamps=True,
                            **add_electrical_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_series_kwargs["es_key"])
                    else:
                        # always add recording electrodes, as they will be used by Units
                        add_electrodes_to_nwbfile(recording=recording, nwbfile=nwbfile, metadata=electrode_metadata)

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

                        if recording_lfp is None:
                            # Wide-band recording: filter and resample LFP
                            print(
                                f"\tAdding LFP data for stream {stream_name} from wide-band signal - segment {segment_index}"
                            )
                            recording_lfp = spre.bandpass_filter(recording, **lfp_filter_kwargs)
                            recording_lfp = spre.resample(recording_lfp, lfp_sampling_rate)
                            recording_lfp = spre.astype(recording_lfp, dtype="int16")

                            # there is a bug in with sample mismatches for the last chunk if num_samples not divisible by chunk_size
                            # the workaround is to discard the last samples to make it "even"
                            if recording.get_num_segments() == 1:
                                recording_lfp = recording_lfp.frame_slice(
                                    start_frame=0,
                                    end_frame=int(
                                        recording_lfp.get_num_samples() // lfp_sampling_rate * lfp_sampling_rate
                                    ),
                                )
                            # set times
                            lfp_period = 1.0 / lfp_sampling_rate
                            for sg_idx in range(recording.get_num_segments()):
                                ts_lfp = (
                                    np.arange(recording_lfp.get_num_samples(sg_idx))
                                    / recording_lfp.sampling_frequency
                                    - recording.get_times(sg_idx)[0]
                                    + lfp_period / 2
                                )
                                recording_lfp.set_times(ts_lfp, segment_index=sg_idx, with_warning=False)
                            save_to_binary = True
                        else:
                            print(f"\tAdding LFP data for {stream_name} from LFP stream - segment {segment_index}")
                            save_to_binary = False
                        channel_ids = recording_lfp.get_channel_ids()

                        # re-reference only for agar - subtract median of channels out of brain using surface channel index arg
                        # similar processing to allensdk
                        if SURFACE_CHANNEL_AGAR_PROBES_INDICES is not None:
                            if probe_device_name in SURFACE_CHANNEL_AGAR_PROBES_INDICES:
                                print(f"\t\tCommon median referencing for probe {probe_device_name}")
                                surface_channel_index = SURFACE_CHANNEL_AGAR_PROBES_INDICES[probe_device_name]
                                # get indices of channels out of brain including surface channel
                                reference_channel_indices = np.arange(surface_channel_index, len(channel_ids))
                                reference_channel_ids = channel_ids[reference_channel_indices]
                                # common median reference to channels out of brain
                                recording_lfp = spre.common_reference(
                                    recording_lfp,
                                    reference="global",
                                    ref_channel_ids=reference_channel_ids,
                                )
                            else:
                                print(f"Could not find {probe_device_name} in surface channel dictionary")

                        # spatial subsampling from allensdk - keep every nth channel
                        if SPATIAL_CHANNEL_SUBSAMPLING_FACTOR > 1:
                            print(f"\t\tSpatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
                            channel_ids_to_keep = channel_ids[0 : len(channel_ids) : SPATIAL_CHANNEL_SUBSAMPLING_FACTOR]
                            recording_lfp = recording_lfp.channel_slice(channel_ids_to_keep)

                        # time subsampling/decimate
                        if TEMPORAL_SUBSAMPLING_FACTOR > 1:
                            print(f"\t\tTemporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
                            recording_lfp_sub = spre.decimate(recording_lfp, TEMPORAL_SUBSAMPLING_FACTOR)
                            for sg_idx in range(recording.get_num_segments()):
                                lfp_times = recording_lfp.get_times(segment_index=sg_idx)
                                recording_lfp_sub.set_times(lfp_times[::TEMPORAL_SUBSAMPLING_FACTOR], segment_index=sg_idx, with_warning=False)
                            recording_lfp = recording_lfp_sub

                        # high pass filter from allensdk
                        if HIGHPASS_FILTER_FREQ_MIN > 0:
                            print(f"\t\tHighpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
                            recording_lfp = spre.highpass_filter(recording_lfp, freq_min=HIGHPASS_FILTER_FREQ_MIN)

                        # Assign to the correct channel group
                        recording_lfp.set_channel_groups([probe_device_name] * recording_lfp.get_num_channels())

                        if STUB_TEST:
                            end_frame = int(STUB_SECONDS * recording_lfp.sampling_frequency)
                            recording_lfp = recording_lfp.frame_slice(start_frame=0, end_frame=end_frame)

                        # For streams without a separate LFP, save to binary to speed up conversion later
                        if save_to_binary:
                            print(f"\tSaving preprocessed LFP to binary")
                            recording_lfp = recording_lfp.save(
                                folder=scratch_folder / f"{recording_name}-LFP", verbose=False
                            )

                        print(f"\tAdding LFP recording {recording_lfp}")
                        add_recording_to_nwbfile(
                            recording=recording_lfp,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            compression=None,
                            always_write_timestamps=True,
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
                configure_backend(nwbfile=nwbfile, backend_configuration=backend_configuration)

                print(f"Writing NWB file to {nwbfile_output_path}")
                if NWB_BACKEND == "zarr":
                    write_args = {"link_data": False}
                    # TODO: enable parallel write for Zarr
                    # write_args = {"number_of_jobs": n_jobs}
                else:
                    write_args = {}

                t_write_start = time.perf_counter()
                with io_class(str(nwbfile_output_path), "w") as export_io:
                    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                t_write_end = time.perf_counter()
                elapsed_time_write = np.round(t_write_end - t_write_start, 2)
                print(f"Writing time: {elapsed_time_write}s")
                print(f"Done writing {nwbfile_output_path}")
                nwb_output_files.append(nwbfile_output_path)

    t_export_end = time.perf_counter()
    elapsed_time_export = np.round(t_export_end - t_export_start, 2)
    print(f"NWB EXPORT ECEPHYS time: {elapsed_time_export}s")
