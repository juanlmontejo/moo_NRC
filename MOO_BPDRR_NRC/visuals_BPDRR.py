import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def best_solution(
    data,
    generations=None,
    pareto_objectives=None
):
    """
    Identify the best compromise solution among the non-dominated
    solutions using the normalized Euclidean distance to the ideal point.

    All objectives are assumed to be minimised.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the optimisation objective values.
        For historical results, the DataFrame should have a MultiIndex
        with levels "Generation" and "Solution".

    generations : list, optional
        Generations to consider when identifying the best solution.
        If None, all generations are considered.

    pareto_objectives : list or tuple, optional
        Objectives used to identify non-dominated solutions and calculate
        the distance to the ideal point.
        If None, all six objectives are used.

    Returns
    -------
    best_solution_id : tuple
        Original (Generation, Solution) identifier.

    best_objectives : pandas.Series
        Objective-function values of the best solution.

    best_distance : float
        Normalized Euclidean distance from the best solution to the ideal point.
    """

    # All objective functions
    objective_names = [
        "FH",
        "t95",
        "RL",
        "TNS",
        "NWSECH",
        "WL"
    ]

    # Use all six objectives by default
    if pareto_objectives is None:
        pareto_objectives = objective_names

    if not isinstance(pareto_objectives, (list, tuple)):
        raise TypeError(
            "pareto_objectives must be a list or tuple of objective names."
        )

    # Check objective names
    unknown_objectives = set(pareto_objectives).difference(
        objective_names
    )

    if unknown_objectives:
        raise ValueError(
            f"Unknown objective name(s): {sorted(unknown_objectives)}"
        )

    # Select generations
    if generations is not None:

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(
                "The DataFrame must have a MultiIndex with "
                "'Generation' and 'Solution' levels when "
                "using the generations argument."
            )

        if "Generation" not in data.index.names:
            raise ValueError(
                "The DataFrame MultiIndex must contain a "
                "'Generation' level."
            )

        available_generations = (
            data.index
            .get_level_values("Generation")
            .unique()
        )

        missing_generations = set(generations).difference(
            available_generations
        )

        if missing_generations:
            raise ValueError(
                f"Generation(s) not found in DataFrame: "
                f"{sorted(missing_generations)}"
            )

        data_selected = data.loc[
            data.index.get_level_values("Generation").isin(generations)
        ]

    else:

        data_selected = data

    # Remove solutions with missing objective values
    data_selected = data_selected.dropna(
        subset=list(pareto_objectives)
    )

    # Objective values
    objective_values = data_selected[
        list(pareto_objectives)
    ].to_numpy(dtype=float)

    # Identify non-dominated solutions
    non_dominated_idx = NonDominatedSorting().do(
        objective_values,
        only_non_dominated_front=True
    )

    # Objective values of non-dominated solutions
    nd_objectives = objective_values[non_dominated_idx]

    # Ideal and worst values
    ideal_values = objective_values.min(axis=0)
    worst_values = objective_values.max(axis=0)

    # Normalize objective values to [0, 1]
    objective_range = worst_values - ideal_values

    # Avoid division by zero if an objective has the same
    # value for every solution
    objective_range[objective_range == 0] = 1

    nd_normalized = (
        nd_objectives - ideal_values
    ) / objective_range

    # Euclidean distance to the ideal point
    distances_to_ideal = np.linalg.norm(
        nd_normalized,
        axis=1
    )

    # Find the closest non-dominated solution
    best_nd_position = np.argmin(
        distances_to_ideal
    )

    # Position of the best solution within data_selected
    best_position = non_dominated_idx[
        best_nd_position
    ]

    # Original (Generation, Solution) identifier
    best_solution_id = data_selected.index[
        best_position
    ]

    # Objective values of the best solution
    best_objectives = data_selected.loc[
        best_solution_id,
        list(pareto_objectives)
    ]

    # Distance to ideal point
    best_distance = distances_to_ideal[
        best_nd_position
    ]

    return (
        best_solution_id,
        best_objectives,
        best_distance
    )

# Function to plot a Pareto front with four objectives, using position, colour, and marker size.
# This function also allows to choose multiple generations to plot from the MOO proccess.
def plot_pareto_OF_MG(
    data,
    obj1,
    obj2,
    color_obj,
    size_obj,
    plot_info=None,
    generations=None,
    pareto_objectives=None
):
    """
    Plot a Pareto front with four objectives using position,
    colour, and marker size.

    If `data` has a MultiIndex with levels "Generation" and
    "Solution", selected generations can be plotted using
    the `generations` argument.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the optimisation objective values.

        For historical results, the DataFrame should have a
        MultiIndex with levels:
            "Generation"
            "Solution"

    obj1 : str
        Objective plotted on the x-axis.

    obj2 : str
        Objective plotted on the y-axis.

    color_obj : str
        Objective represented by colour.

    size_obj : str
        Objective represented by marker size.

    plot_info : dict, optional
        Metadata used to label the figure. Expected keys:
            "scenario"
            "solution"
            "crews"

    generations : list, optional
        Generations to plot.

        Example:
            generations=[46, 47, 48, 49, 50]

        If None, all generations in the DataFrame are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Generated axes.
    """

    # Check objective names
    objective_names = [
        "FH",
        "t95",
        "RL",
        "TNS",
        "NWSECH",
        "WL"
    ]

    selected_names = [
        obj1,
        obj2,
        color_obj,
        size_obj
    ]

    unknown = set(selected_names).difference(objective_names)

    if unknown:
        raise ValueError(
            f"Unknown objective name(s): {sorted(unknown)}"
        )

    if len(set(selected_names)) != 4:
        raise ValueError(
            "obj1, obj2, color_obj and size_obj "
            "must be different objectives."
        )

    # Objective units
    objective_units = {
        "FH": "min",
        "t95": "min",
        "RL": "%-min",
        "TNS": "min",
        "NWSECH": "",
        "WL": "m³"
    }

    def objective_label(name):
        unit = objective_units[name]

        if unit:
            return f"{name} ({unit})"
        else:
            return name
    
    # Check DataFrame columns
    missing_columns = set(selected_names).difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing objective column(s): "
            f"{sorted(missing_columns)}"
        )

    # Select generations
    if generations is not None:

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(
                "The DataFrame must have a MultiIndex with "
                "'Generation' and 'Solution' levels when "
                "using the generations argument."
            )

        if "Generation" not in data.index.names:
            raise ValueError(
                "The DataFrame MultiIndex must contain a "
                "'Generation' level."
            )

        available_generations = (
            data.index.get_level_values("Generation")
            .unique()
        )

        missing_generations = set(generations).difference(
            available_generations
        )

        if missing_generations:
            raise ValueError(
                f"Generation(s) not found in DataFrame: "
                f"{sorted(missing_generations)}"
            )

        data_plot = data.loc[
            data.index.get_level_values("Generation").isin(
                generations
            )
        ]

    else:
        data_plot = data

    # Extract objective values
    x_values = data_plot[obj1].to_numpy(dtype=float)
    y_values = data_plot[obj2].to_numpy(dtype=float)
    colour_values = data_plot[color_obj].to_numpy(dtype=float)
    size_values = data_plot[size_obj].to_numpy(dtype=float)

    # # Objectives used to identify non-dominated solutions
    # if pareto_objectives is None:
    #     pareto_objectives = objective_names

    # if not isinstance(pareto_objectives, (list, tuple)):
    #     raise TypeError(
    #         "pareto_objectives must be a list or tuple of objective names."
    #     )

    # if len(pareto_objectives) < 2:
    #     raise ValueError(
    #         "At least two objectives are required for non-dominated sorting."
    #     )

    # unknown_pareto_objectives = set(pareto_objectives).difference(
    #     objective_names
    # )

    # if unknown_pareto_objectives:
    #     raise ValueError(
    #         "Unknown objective name(s) in pareto_objectives: "
    #         f"{sorted(unknown_pareto_objectives)}"
    #     )

    # missing_pareto_columns = set(pareto_objectives).difference(
    #     data_plot.columns
    # )

    # if missing_pareto_columns:
    #     raise ValueError(
    #         "Missing Pareto objective column(s): "
    #         f"{sorted(missing_pareto_columns)}"
    #     )

    # objective_values = data_plot[
    #     list(pareto_objectives)
    # ].to_numpy(dtype=float)

    # non_dominated_idx = NonDominatedSorting().do(
    #     objective_values,
    #     only_non_dominated_front=True
    # )

    # Find the best compromise solution
    best_solution_id, best_objectives, best_distance = best_solution(
        data,
        generations=generations,
        pareto_objectives=pareto_objectives
    )

    # Get its position within the plotted data
    best_solution_idx = data_plot.index.get_loc(
        best_solution_id
    )

    # Scale marker sizes
    size_min = size_values.min()
    size_max = size_values.max()

    min_marker_size = 20
    max_marker_size = 200

    if size_max == size_min:

        marker_sizes = np.full(
            len(size_values),
            (min_marker_size + max_marker_size) / 2
        )

    else:

        # Inverted relationship:
        # low objective value = large marker
        # high objective value = small marker

        marker_sizes = (
            max_marker_size - ((size_values - size_min)/ (size_max - size_min)) 
            * (max_marker_size - min_marker_size)
        )

    # Create figure
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # Plot Pareto front
    scatter = ax.scatter(
        x_values,
        y_values,
        c=colour_values,
        s=marker_sizes,
        cmap="viridis_r",
        alpha=0.8
    )

    # # Highlight non-dominated solutions
    # ax.scatter(
    #     x_values[non_dominated_idx],
    #     y_values[non_dominated_idx],
    #     s=marker_sizes[non_dominated_idx] + 40,
    #     facecolors="none",
    #     edgecolors="black",
    #     linewidths=1.8,
    #     label="Non-dominated solution"
    # )

    # Highlight the best compromise solution
    ax.scatter(
        x_values[best_solution_idx],
        y_values[best_solution_idx],
        marker="*",
        s=250,
        color="crimson",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
        label="Best compromise solution"
    )

    ax.set_xlabel(objective_label(obj1))
    ax.set_ylabel(objective_label(obj2))

    # Ideal point
    ideal_x = data_plot[obj1].min()
    ideal_y = data_plot[obj2].min()


    ax.scatter(
        ideal_x,
        ideal_y,
        marker="X",
        s=100,
        color="black",
        linewidths=1.5
    )

    # Build title
    title = (
        f"Pareto front: {obj2} vs {obj1}"
        f"\ncolour = {color_obj}, size = {size_obj}"
    )

    if plot_info is not None:

        scenario = plot_info.get("scenario")
        crews = plot_info.get("crews")

        info = []

        if scenario is not None:
            info.append(str(scenario))

        if crews is not None:
            info.append(f"{crews} crews")

        if info:
            title += "\n" + " | ".join(info)

    # Add generation information
    if generations is not None:
        title += (
            "\nGenerations: "
            + ", ".join(str(g) for g in generations)
        )

    ax.set_title(title)

    # Colour bar
    cbar = fig.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label(objective_label(color_obj))

    # Legend handles
    # Ideal point
    ideal_handle = Line2D(
        [],
        [],
        marker="X",
        linestyle="None",
        markersize=10,
        markeredgewidth=1.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label="Ideal point"
    )

    # # Non-dominated solution
    # non_dominated_handle = Line2D(
    #     [],
    #     [],
    #     marker="o",
    #     linestyle="None",
    #     markersize=9,
    #     markerfacecolor="none",
    #     markeredgecolor="black",
    #     markeredgewidth=1.8,
    #     label="Non-dominated solution"
    # )

    # Best solution
    best_handle = Line2D(
        [],
        [],
        marker="*",
        linestyle="None",
        markersize=13,
        markerfacecolor="crimson",
        markeredgecolor="black",
        markeredgewidth=1.2,
        label="Best solution"
    )

    # Create legend
    ideal_legend = ax.legend(
        handles=[
            ideal_handle,
            # non_dominated_handle,
            best_handle
        ],
        loc="upper right",
        frameon=True
    )

    # Marker-size legend
    size_legend_values = np.linspace(
        size_max,
        size_min,
        3
    )

    size_legend_handles = []

    for value in size_legend_values:

        if size_max == size_min:

            marker_size = (
                min_marker_size + max_marker_size
            ) / 2

        else:

            marker_size = (
                max_marker_size
                - (
                    (value - size_min)
                    / (size_max - size_min)
                ) * (
                    max_marker_size - min_marker_size
                )
            )

        handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=np.sqrt(marker_size),
            markerfacecolor="gray",
            markeredgecolor="gray",
            alpha=0.6,
            label=f"{value:.0f}"
        )

        size_legend_handles.append(handle)

    size_legend = ax.legend(
        handles=size_legend_handles,
        title=objective_label(size_obj),
        loc="center right",
        frameon=True
    )

    # Keep both legends
    ax.add_artist(ideal_legend)

    # Grid and layout
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    plt.show()

    return fig, ax

# Function to plot a Pareto front with four objectives, using position, colour, and marker size.
# This function is meant to be used for the older results of the research since it can't choose
# one or multiple generations to plot. It only takes the last generation from the MOO.
def plot_pareto_4_OF(
    data,
    obj1,
    obj2,
    color_obj,
    size_obj,
    plot_info=None
):
    """
    Plot two objectives as a Pareto front, with a third objective
    represented by colour and a fourth objective represented by
    marker size.

    The marker size is inversely proportional to the size objective:
    higher objective values are represented by smaller markers.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the optimisation objective values.

    obj1 : str
        Objective plotted on the x-axis.

    obj2 : str
        Objective plotted on the y-axis.

    color_obj : str
        Objective represented by the colour of each point.

    size_obj : str
        Objective represented by the size of each point.

    plot_info : dict, optional
        Metadata used to label the figure. Expected keys are:
            "scenario"
            "solution"
            "crews"

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Generated axes.
    """

    # Check objective names
    objective_names = [
        "FH",
        "t95",
        "RL",
        "TNS",
        "NWSECH",
        "WL"
    ]

    selected_names = [
        obj1,
        obj2,
        color_obj,
        size_obj
    ]

    unknown = set(selected_names).difference(objective_names)

    if unknown:
        raise ValueError(
            f"Unknown objective name(s): {sorted(unknown)}"
        )

    if len(set(selected_names)) != 4:
        raise ValueError(
            "obj1, obj2, color_obj and size_obj "
            "must be different objectives."
        )

    # Check DataFrame columns
    missing_columns = set(selected_names).difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing objective column(s): "
            f"{sorted(missing_columns)}"
        )

    # Extract objective values
    x_values = data[obj1].to_numpy(dtype=float)
    y_values = data[obj2].to_numpy(dtype=float)
    colour_values = data[color_obj].to_numpy(dtype=float)
    size_values = data[size_obj].to_numpy(dtype=float)

    # Scale marker sizes
    size_min = size_values.min()
    size_max = size_values.max()

    min_marker_size = 40
    max_marker_size = 200

    if size_max == size_min:

        marker_sizes = np.full(
            len(size_values),
            (min_marker_size + max_marker_size) / 2
        )

    else:

        # Inverted relationship:
        # low objective value= large marker
        # high objective value= small marker

        marker_sizes = (
            max_marker_size
            - (
                (size_values - size_min)
                / (size_max - size_min)
            ) * (
                max_marker_size - min_marker_size
            )
        )

    # Create figure
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # Plot Pareto front
    scatter = ax.scatter(
        x_values,
        y_values,
        c=colour_values,
        s=marker_sizes,
        cmap="viridis_r",
        alpha=0.8
    )

    ax.set_xlabel(obj1)
    ax.set_ylabel(obj2)

    # Ideal point
    ideal_x = data[obj1].min()
    ideal_y = data[obj2].min()

    ideal_point = ax.scatter(
        ideal_x,
        ideal_y,
        marker="X",
        s=100,
        color="black",
        linewidths=1.5
    )

    # Build title using plot_info
    title = (
        f"Pareto front: {obj2} vs {obj1}"
        f"\ncolour = {color_obj}, size = {size_obj}"
    )

    if plot_info is not None:

        scenario = plot_info.get("scenario")
#        solution = plot_info.get("solution")
        crews = plot_info.get("crews")

        info = []

        if scenario is not None:
            info.append(str(scenario))

#        if solution is not None:
#            info.append(f"Solution {solution}")

        if crews is not None:
            info.append(f"{crews} crews")

        if info:
            title += "\n" + " | ".join(info)

    ax.set_title(title)

    # Colour bar
    cbar = fig.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label(color_obj)

    # Ideal point legend
    ideal_handle = Line2D(
        [],
        [],
        marker="X",
        linestyle="None",
        markersize=10,
        markeredgewidth=1.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label="Ideal point"
    )

    ideal_legend = ax.legend(
        handles=[ideal_handle],
        loc="upper right",
        frameon=True
    )


    # Marker-size legend
    # Representative values, from HIGH to LOW
    size_legend_values = np.linspace(
        size_max,
        size_min,
        3
    )

    size_legend_handles = []

    for value in size_legend_values:

        if size_max == size_min:

            marker_size = (
                min_marker_size + max_marker_size
            ) / 2

        else:

            marker_size = (
                max_marker_size
                - (
                    (value - size_min)
                    / (size_max - size_min)
                ) * (
                    max_marker_size - min_marker_size
                )
            )

        handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=np.sqrt(marker_size),
            markerfacecolor="gray",
            markeredgecolor="gray",
            alpha=0.6,
            label=f"{value:.0f}"
        )

        size_legend_handles.append(handle)

    size_legend = ax.legend(
        handles=size_legend_handles,
        title=size_obj,
        loc="center right",
        frameon=True
    )

    # Keep both legends
    ax.add_artist(ideal_legend)

    # Grid and layout
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    plt.show()

    return fig, ax

# Function to plot system functionality over the restoration period.
def plot_functionality(
    functionality_series,
    t95=None,
    scenario_name=None,
    sol_sel=None,
    y_min=75,
    y_max=105,
):
    """
    Plot system functionality over the restoration period.

    Parameters
    ----------
    functionality_series : pandas.Series
        System functionality over simulation time.

    t95 : float, optional
        Stable recovery time in minutes.

    scenario_name : str, optional
        Name of the damage scenario.

    y_min, y_max : float
        Limits for the functionality axis.
    """

    time_hours = functionality_series.index.to_numpy() / 3600
    functionality = functionality_series.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        time_hours,
        functionality,
        linewidth=2,
        label="System functionality"
    )

    ax.axhline(
        100,
        linestyle="--",
        linewidth=1,
        label="Full functionality"
    )

    ax.axhline(
        95,
        linestyle=":",
        linewidth=1.5,
        label="95% threshold"
    )

    # Resilience loss area
    ax.fill_between(
        time_hours,
        functionality,
        100,
        alpha=0.15,
        label="Resilience loss"
    )

    # t95
    if t95 is not None:
        t95_hours = t95 / 60

        ax.axvline(
            t95_hours,
            linestyle="--",
            linewidth=1.5
        )

        ax.annotate(
            f"t95 = {t95_hours:.1f} h",
            xy=(t95_hours, 95),
            xytext=(6, 8),
            textcoords="offset points"
        )

    # Labels and title
    title = "System functionality"

    if scenario_name is not None:
        title += f" - {scenario_name} - solution {sol_sel}"

    ax.set_title(title)
    ax.set_xlabel("Time after earthquake [hours]")
    ax.set_ylabel("Functionality [%]")
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()

    return fig, ax

# Function to plot water loss during the restoration period.
def plot_water_loss(
    demand,
    emitter_nodes,
    dt,
    scenario_name=None,
    sol_sel=None,
    duration_days=None
):
    """
    Plot water loss per simulation timestep.

    Parameters
    ----------
    demand : pandas.DataFrame
        Simulated node demand in L/s.

    emitter_nodes : list
        IDs of emitter/leak nodes.

    dt : float
        Hydraulic timestep in seconds.

    scenario_name : str, optional
        Name of the damage scenario.

    duration_days : float, optional
        Simulation duration in days.
    """

    # Leakage flow at each timestep
    leakage = demand[emitter_nodes]

    leakage_flow_ts = leakage.sum(axis=1)

    # Water loss during each timestep [m3]
    water_loss_ts = (
        leakage_flow_ts * dt
    ) / 1000

    # Simulation time [days]
    time_days = (
        water_loss_ts.index / (24 * 3600)
    )

    # Plot
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        time_days,
        water_loss_ts,
        marker="o",
        linewidth=2
    )

    # Labels and title
    ax.set_xlabel("Simulation time (days)")
    ax.set_ylabel("Water loss per timestep (m³)")

    title = "Water Loss During Restoration"

    if scenario_name is not None:
        title += f" - {scenario_name} - solution {sol_sel}"

    ax.set_title(title)

    ax.grid(True)

    if duration_days is not None:
        ax.set_xlim(0, duration_days)

    fig.tight_layout()

    plt.show()

    return fig, ax