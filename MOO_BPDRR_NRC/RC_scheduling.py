# This module contains functions for the repair crew scheduling and the generation of EPANET control statements.

import pandas as pd
import os

# Function to create a repair schedule from a permutation of damaged pipes.
# This version of the function has NO BREAK TIME for the repair crews, which means they work continously
def create_schedule_NBT(
    reparations,
    dmatrix,
    pipe_ids,
    indexes,
    n_crews=3,
    initial_travel_time=0.75,  #This value considers both preparation time (0.5) and initial time travel (0.25)
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
        Preparation and travel time assigned to the first repair of each crew.

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

# Function that limits the repair crews to work and travel only between working hours
def add_working_time(
    start_time,
    duration,
    work_start=7.0,
    work_end=17.0,
    start_clock_hour=7.0,
    daily_travel_time=0.25,
    include_daily_travel=False
):
    """
    Add an activity duration while allowing work only during
    working hours.

    When an activity continues into a new working day, the crew
    can optionally spend daily_travel_time travelling from the
    operation center to the location where it will continue working.

    Parameters
    ----------
    start_time : float
        Starting simulation time.

    duration : float
        Duration of the activity in working hours.

    work_start : float, optional
        Start of the working day.
        Default: 07:00.

    work_end : float, optional
        End of the working day.
        Default: 17:00.

    start_clock_hour : float, optional
        Clock time corresponding to simulation time 0.
        Default: 07:00.

    daily_travel_time : float, optional
        Travel time from the operation center to the crew's
        work location at the beginning of a new working day.
        Default: 0.25 hours.

    include_daily_travel : bool, optional
        If True, daily_travel_time is added whenever the activity
        continues into a new working day.
        Default: False.

    Returns
    -------
    float
        Elapsed simulation time at the end of the activity.
    """

    current_time = start_time
    remaining_time = duration

    while True:

        # Convert simulation time into actual clock time
        hour_of_day = (
            current_time + start_clock_hour
        ) % 24

        # Before working hours
        if hour_of_day < work_start:
            current_time += (
                work_start - hour_of_day
            )

        # At or after the end of working hours
        elif hour_of_day >= work_end:
            current_time += (
                (24 - hour_of_day) + work_start
            )

            # Crew starts the new working day by travelling
            # from the operation center to its work location.
            if include_daily_travel:
                current_time += daily_travel_time

        # No duration remaining
        if remaining_time <= 0:
            return current_time

        # Available working time until the end of today
        hour_of_day = (
            current_time + start_clock_hour
        ) % 24

        available_time = work_end - hour_of_day

        # Activity finishes during this working period
        if remaining_time <= available_time:
            current_time += remaining_time
            return current_time

        # Activity continues beyond today's working hours
        current_time += available_time
        remaining_time -= available_time

        # Move to the next working day at work_start
        current_time += (
            (24 - work_end) + work_start
        )

        # If this activity continues into the next day,
        # the crew must travel from the operation center
        # to the location where the activity will continue.
        if include_daily_travel:
            current_time += daily_travel_time

# Function to create a repair schedule from a permutation of damaged pipes.
# This version of the function add break times to the repair crews by using the add_working_time() function
def create_schedule(
    reparations,
    dmatrix,
    pipe_ids,
    indexes,
    n_crews=3,
    preparation_time=0.5,
    initial_travel_time=0.25,
    work_start=7.0,
    work_end=17.0,
    start_clock_hour=7.0,
    verbose=False
):
    """
    Creates a repair schedule from a permutation of damaged pipes.

    Crews work only between work_start and work_end.

    At the beginning of the first working day, each crew:
        1. completes preparation_time
        2. travels from the operation center to its first pipe
           using initial_travel_time

    At the beginning of every subsequent working day, each crew:
        1. starts from the operation center
        2. travels to the last pipe where it worked the previous day
           using initial_travel_time
        3. then travels from that pipe to its next assigned pipe
           using dmatrix

    Parameters
    ----------
    reparations : dict
        Dictionary containing repair information.

    dmatrix : pandas.DataFrame
        Square travel-time matrix between damaged pipes.

    pipe_ids : list
        Ordered list of damaged pipe IDs.

    indexes : list
        Permutation of integer indices defining repair priority.

    n_crews : int, optional
        Number of repair crews. Default: 3.

    preparation_time : float, optional
        Preparation time before the first deployment.
        Default: 0.5 hours.

    initial_travel_time : float, optional
        Assumed travel time between the operation center and
        the crew's work location at the beginning of a working day.
        Default: 0.25 hours.

    work_start : float, optional
        Start of the working period.
        Default: 7.0.

    work_end : float, optional
        End of the working period.
        Default: 17.0.

    start_clock_hour : float, optional
        Clock time corresponding to simulation time 0.
        Default: 7.0.

    verbose : bool, optional
        Print scheduling information.
        Default: False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        Order, Pipe, Crew, Travel, Repair, Start, Finish.
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
            "available_at": preparation_time,
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

        available_at = crews[crew]["available_at"]
        previous_pipe = crews[crew]["last_pipe"]

        # Determine whether crew is starting a new working day
        # Clock time of the crew's current availability
        current_clock_hour = (
            start_clock_hour
            + available_at
        ) % 24

        # A new working day starts when the crew is available
        # before the working period begins.
        new_working_day = (
            previous_pipe is not None
            and current_clock_hour <= work_start
        )

        # Determine travel time
        if previous_pipe is None:

            # First assignment of the crew
            #
            # The crew starts from the assumed operation center.
            travel_time = initial_travel_time

        elif new_working_day:
            # Crew is starting a new working day.
            #
            # It first travels from the operation center to the
            # location where it worked the previous day,
            # then travels from the previous pipe to the next pipe.
            travel_time = (
                initial_travel_time
                + dmatrix.loc[
                    previous_pipe,
                    pipe_id
                ]
            )

        else:

            # Same working day:
            # travel directly from the previous repaired pipe
            # to the next assigned pipe.
            travel_time = dmatrix.loc[
                previous_pipe,
                pipe_id
            ]

        # Repair time
        repair_time = reparations[pipe_id]["t_r"]

        # Travel to damaged pipe
        arrival_time = add_working_time(
            crews[crew]["available_at"],
            travel_time,
            work_start=work_start,
            work_end=work_end,
            start_clock_hour=start_clock_hour,
            include_daily_travel=False
        )

        # Repair starts after travel
        start_time = arrival_time

        # Repair takes place only during working hours
        finish_time = add_working_time(
            start_time,
            repair_time,
            work_start=work_start,
            work_end=work_end,
            start_clock_hour=start_clock_hour,
            daily_travel_time=initial_travel_time,
            include_daily_travel=True
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
                f"Travel {travel_time:5.2f} | "
                f"Start {start_time:7.2f} | "
                f"Finish {finish_time:7.2f}"
            )

    # Build schedule DataFrame
    schedule = pd.DataFrame(schedule_rows)

    # Chronological order
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
