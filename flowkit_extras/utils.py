import os
import re
import pandas as pd
import flowkit as fk


def extract_gated_data(
        output_dir,
        wsp_file,
        fcs_dir,
        sample_group,
        event_columns=None,
        exclude_samples=None,
        verbose=False
):
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

        # TODO: save events_df & gates_df to feather here
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
