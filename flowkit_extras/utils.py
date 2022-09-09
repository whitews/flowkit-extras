"""
Utility functions
"""
import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import flowkit as fk


def _process_bead_samples(bead_samples):
    # do nothing if there are no bead samples
    bead_sample_count = len(bead_samples)
    if bead_sample_count == 0:
        warnings.warn("No bead samples were loaded")
        return

    bead_lut = {}

    # all the bead samples must have the same panel, use the 1st one to
    # determine the fluorescence channels
    fluoro_indices = bead_samples[0].fluoro_indices

    # 1st check is to make sure the # of bead samples matches the #
    # of fluorescence channels
    if bead_sample_count != len(fluoro_indices):
        raise ValueError("Number of bead samples must match the number of fluorescence channels")

    # get PnN channel names from 1st bead sample
    pnn_labels = []
    for f_idx in fluoro_indices:
        pnn_label = bead_samples[0].pnn_labels[f_idx]
        if pnn_label not in pnn_labels:
            pnn_labels.append(pnn_label)
            bead_lut[f_idx] = {'pnn_label': pnn_label}
        else:
            raise ValueError("Duplicate channel labels are not supported")

    # now, determine which bead file goes with which channel, and make sure
    # they all have the same channels
    for i, bs in enumerate(bead_samples):
        # check file name for a match with a channel
        if bs.fluoro_indices != fluoro_indices:
            raise ValueError("All bead samples must have the same channel labels")

        for channel_idx, lut in bead_lut.items():
            # file names typically don't have the "-A", "-H', or "-W" sub-strings
            pnn_label = lut['pnn_label'].replace("-A", "")

            if pnn_label in bs.original_filename:
                lut['bead_index'] = i
                lut['pns_label'] = bs.pns_labels[channel_idx]

    return bead_lut


def calculate_compensation_from_beads(comp_bead_samples, matrix_id='comp_bead'):
    """
    Calculates spillover from a list of FCS bead files.

    :param comp_bead_samples: str or list. If given a string, it can be a directory path or a file path.
        If a directory, any .fcs files in the directory will be loaded. If a list, then it must
        be a list of file paths or a list of Sample instances. Lists of mixed types are not
        supported.
    :param matrix_id: label for the calculated Matrix
    :return: a Matrix instance
    """
    bead_samples = fk.load_samples(comp_bead_samples)
    bead_lut = _process_bead_samples(bead_samples)
    if len(bead_lut) == 0:
        warnings.warn("No bead samples were loaded")
        return

    detectors = []
    fluorochromes = []
    comp_values = []
    for channel_idx in sorted(bead_lut.keys()):
        detectors.append(bead_lut[channel_idx]['pnn_label'])
        fluorochromes.append(bead_lut[channel_idx]['pns_label'])
        bead_idx = bead_lut[channel_idx]['bead_index']

        x = bead_samples[bead_idx].get_events(source='raw')[:, channel_idx]
        good_events = x < (2 ** 18) - 1
        x = x[good_events]

        comp_row_values = []
        for channel_idx2 in sorted(bead_lut.keys()):
            if channel_idx == channel_idx2:
                comp_row_values.append(1.0)
            else:
                y = bead_samples[bead_idx].get_events(source='raw')[:, channel_idx2]
                y = y[good_events]
                rlm_res = sm.RLM(y, x).fit()

                # noinspection PyUnresolvedReferences
                comp_row_values.append(rlm_res.params[0])

        comp_values.append(comp_row_values)

    return fk.Matrix(matrix_id, np.array(comp_values), detectors, fluorochromes)


def extract_gated_data(
        output_dir,
        wsp_file,
        fcs_dir,
        sample_group,
        event_columns=None,
        exclude_samples=None,
        verbose=False
):
    """
    Extracts pre-processed event data and gate membership for each of the FCS
    files referenced in given FlowJo10 workspace file (wsp file). Both event
    and gate membership data are saved as Feather files to the provided output
    directory. Two subdirectories, 'events' and 'gates', will be created in
    the output directory where 2 Feather files will be created for each FCS
    file, one in the 'events' directory (prepended with 'events_') and one in
    the 'gates' directory (prepended with 'gates_').

    :param output_dir: File path location for output Feather files
    :param wsp_file: File path of FlowJo 10 workspace (wsp file)
    :param fcs_dir: Directory path containing FCS files
    :param sample_group: Sample group label within the WSP file from which
        gated events and gate membership data will be extracted
    :param event_columns: List of channels/markers to include in the
        preprocessed event data. If None, all channels are exported.
    :param exclude_samples: List of FCS sample IDs to exclude from export
    :param verbose: If True, prints various issues found when parsing
        the workspace and FCS files (missing FCS files, missing channel
        columns, missing gates, and output array shapes)
    :return: None
    """
    fs = fk.Session()
    fs.import_flowjo_workspace(wsp_file, ignore_missing_files=True)

    # find FCS files in given fcs_dir
    fcs_files = [f for f in os.listdir(fcs_dir) if re.search(r'.*\.fcs$', f)]

    sample_ids = fs.get_group_sample_ids(
        sample_group,
        loaded_only=False
    )

    events_dir = os.path.join(output_dir, 'events')
    gates_dir = os.path.join(output_dir, 'gates')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(gates_dir, exist_ok=True)

    sample_paths_to_load = []
    samples_to_process = []

    for f in sample_ids:
        if f in exclude_samples:
            continue

        if f not in fcs_files and verbose:
            print('%s not found in our list' % f)

        sample_paths_to_load.append(os.path.join(fcs_dir, f))
        samples_to_process.append(f)

    fs.add_samples(sample_paths_to_load)
    fs.analyze_samples(sample_group)

    for sample_id in samples_to_process:
        # extract all processed events using gate ID as None
        events_df = fs.get_wsp_gated_events(
            sample_group,
            [sample_id],
            gate_name=None,
            gate_path=None
        )[0]

        if event_columns:
            try:
                events_df = events_df[event_columns]
            except KeyError:
                print(f'Sample to exclude: {sample_id}')
                continue

        # now get gate membership for all gates & all events
        gates_df = pd.DataFrame()
        gate_ids = fs.get_gate_ids(sample_group)
        for gate_name, gate_path in gate_ids:
            try:
                s_gate_membership = fs.get_gate_membership(
                    sample_group,
                    sample_id,
                    gate_name=gate_name,
                    gate_path=gate_path
                )
                gates_df['/'.join([*gate_path, gate_name])] = s_gate_membership
            except KeyError:
                if verbose:
                    print("Gate %s (%s) not found in sample group" % (gate_name, gate_path))

        # add sample_id field to gate DataFrame & put it first
        gates_df['sample_id'] = sample_id
        sample_id_col = gates_df.pop('sample_id')
        gates_df.insert(0, sample_id_col.name, sample_id_col)

        # save events_df & gates_df to feather here
        sample_feather_basename = sample_id.replace('fcs', 'feather')
        events_filename = "events_" + sample_feather_basename
        gates_filename = "gates_" + sample_feather_basename

        events_path = os.path.join(events_dir, events_filename)
        gates_path = os.path.join(gates_dir, gates_filename)

        if not os.path.exists(events_path):
            events_df.reset_index(drop=True, inplace=True)
            events_df.to_feather(events_path)

            print(events_df.shape)

        if not os.path.exists(gates_path):
            gates_df.reset_index(drop=True, inplace=True)
            gates_df.to_feather(gates_path)

            print(gates_df.shape)
