import pandas as pd
import os

# Function to create a repair schedule from a permutation of damaged pipes
def create_schedule(
    reparations,
    dmatrix,
    pipe_ids,
    indexes,
    n_crews=3,
    initial_travel_time=0.5,
    verbose=False
):
    """
    Creates a repair schedule from a permutation of damaged pipes.

    Parameters
    ----------
    reparations : dict
        Dictionary containing repair information.

        Expected format:
        {pipe_id: {"t_r": repair_time},...}

    dmatrix : pandas.DataFrame
        Square travel-time matrix.
        Rows and columns must be indexed by pipe IDs.

    pipe_ids : list
        Ordered list of damaged pipe IDs.

        Example:
        ['153', '742', '981', ...]

    indexes : list
        Permutation of integer indices defining the repair priority of the damaged pipes.

        Example:
        [4, 1, 7, 0, 5, 2, 6, 3]

        This means:

        pipe_ids[4]
        pipe_ids[1]
        pipe_ids[7]
        ...

    n_crews : Number of repair crews (integer).

    initial_travel_time : float, optional
        Travel time assigned to the first repair of each crew.

    verbose : bool, optional
        Print scheduling information.

    Output / Returns
    -------
    Pandas DataFrame with columns: Order, Pipe, Crew, Travel, Repair, Start, Finish.
    """

    # Validate permutation

    n_pipes = len(pipe_ids)

    if len(indexes) != n_pipes:
        raise ValueError(
            "Length of indexes does not match number of damaged pipes."
        )

    if sorted(indexes) != list(range(n_pipes)):
        raise ValueError(
            "indexes must be a permutation of integers "
            "from 0 to len(pipe_ids)-1."
        )

    # Validate dictionaries
    missing_repairs = set(pipe_ids) - set(reparations.keys())

    if missing_repairs:
        raise KeyError(
            f"Missing repair information for pipes: "
            f"{sorted(missing_repairs)}"
        )

    missing_matrix = set(pipe_ids) - set(dmatrix.index)

    if missing_matrix:
        raise KeyError(
            f"Missing pipes from travel-time matrix: "
            f"{sorted(missing_matrix)}"
        )

    # Initialize crews
    crews = {
        crew: {
            "available_at": 0.0,
            "last_pipe": None
        }
        for crew in range(1, n_crews + 1)
    }

    schedule_rows = []

    # Main scheduling loop
    for order, idx in enumerate(indexes, start=1):

        # Convert integer index into actual pipe ID
        pipe_id = pipe_ids[idx]

        # Crew that becomes available first
        crew = min(
            crews,
            key=lambda c: crews[c]["available_at"]
        )

        # Travel time

        previous_pipe = crews[crew]["last_pipe"]

        if previous_pipe is None:

            travel_time = initial_travel_time

        else:

            travel_time = dmatrix.loc[
                previous_pipe,
                pipe_id
            ]

        # Repair time
        repair_time = reparations[pipe_id]["t_r"]

        # Timing

        start_time = (
            crews[crew]["available_at"]
            + travel_time
        )

        finish_time = (
            start_time
            + repair_time
        )

        # Save row

        schedule_rows.append({

            "Order": order,

            "Pipe": pipe_id,

            "Crew": crew,

            "Travel": travel_time,

            "Repair": repair_time,

            "Start": start_time,

            "Finish": finish_time

        })

        # Update crew
        crews[crew]["available_at"] = finish_time

        crews[crew]["last_pipe"] = pipe_id

        if verbose:

            print(
                f"Order {order:3d} | "
                f"Crew {crew:2d} | "
                f"Pipe {pipe_id:<12} | "
                f"Start {start_time:7.2f} | "
                f"Finish {finish_time:7.2f}"
            )

    # Build schedule DataFrame
    schedule = pd.DataFrame(schedule_rows)

    # Chronological order (useful for simulation)
    schedule = (
        schedule
        .sort_values("Start")
        .reset_index(drop=True)
    )

    return schedule


# Function to generate EPANET control statements from the repair schedule
def generate_controls(schedule, comment_lines=True):
    """
    Generate EPANET control statements from a repair schedule.

    Parameters
    ----------
    schedule : pandas.DataFrame
        Repair schedule returned by create_schedule().

        Required columns:
            - Pipe
            - Start
            - Finish

    comment_lines : bool, optional
        If True, insert comment lines identifying each repaired pipe.

    Output / Returns
    -------
    List of EPANET controls.
    """

    # Validate schedule
    required_columns = {"Pipe", "Start", "Finish"}

    missing = required_columns - set(schedule.columns)

    if missing:
        raise ValueError(
            f"Schedule is missing required columns: {sorted(missing)}"
        )

    # Generate controls in chronological order
    schedule_sorted = (
    schedule
    .sort_values("Start")
    .reset_index(drop=True)
)

    new_controls = []

    for _, row in schedule_sorted.iterrows():

        pipe = str(row["Pipe"])
        start = float(row["Start"])
        finish = float(row["Finish"])

        if comment_lines:
            new_controls.append(f"; Pipe {pipe}")

        # Close the two damaged pipe segments
        new_controls.append(
            f"LINK {pipe}_A CLOSED AT TIME {start:.2f}"
        )

        new_controls.append(
            f"LINK {pipe}_B CLOSED AT TIME {start:.2f}"
        )

        # Reopen the original repaired pipe
        new_controls.append(
            f"LINK {pipe} OPEN AT TIME {finish:.2f}"
        )

    return new_controls

# Function to write a new INP file with the generated controls
def write_inp_controls(input_inp, output_inp, schedule, new_controls):
    """
    Creates a new EPANET INP file using the control rules generated by
    generate_controls().

    Parameters
    ----------
    input_inp : str
        Path to the original damaged INP file.

    output_inp : str
        Path where the modified INP file will be saved.

    schedule : pandas.DataFrame
        Repair schedule returned by create_schedule().
        (Currently only stored for compatibility and future extensions.)

    new_controls : list of str
        List of EPANET control lines generated by generate_controls().

    Output / Returns
    -------
    Path to the generated INP file.
    """

    # Basic checks
    if not os.path.exists(input_inp):
        raise FileNotFoundError(f"Input file not found:\n{input_inp}")

    if not isinstance(new_controls, list):
        raise TypeError("new_controls must be a list of strings.")

    if len(new_controls) == 0:
        raise ValueError("new_controls is empty.")

    # Read original INP
    with open(input_inp, "r") as f:
        lines = f.readlines()

    # Locate [CONTROLS]
    controls_start = None

    for i, line in enumerate(lines):
        if line.strip().upper() == "[CONTROLS]":
            controls_start = i
            break

    # If [CONTROLS] does not exist, create it
    if controls_start is None:

        if lines[-1].strip() != "":
            lines.append("\n")

        lines.append("[CONTROLS]\n")
        controls_start = len(lines) - 1

        controls_end = len(lines)

    else:

        # Find the next section
        controls_end = len(lines)

        for j in range(controls_start + 1, len(lines)):

            txt = lines[j].strip()

            if txt.startswith("[") and txt.endswith("]"):
                controls_end = j
                break

    # Build new CONTROLS section
    control_block = [
        "[CONTROLS]\n"
    ]

    for control in new_controls:

        control = control.rstrip()

        if control != "":
            control_block.append(control + "\n")

    control_block.append("\n")

    # Replace old CONTROLS section
    new_lines = (
        lines[:controls_start]
        + control_block
        + lines[controls_end:]
    )

    # Save new INP
    with open(output_inp, "w") as f:
        f.writelines(new_lines)

    return output_inp