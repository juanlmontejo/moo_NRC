import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function to plot a Pareto front with three objectives, using position and colour.
def plot_pareto_3_OF(
    data,
    obj1,
    obj2,
    color_obj,
    plot_info=None
):
    """
    Plot two objectives as a Pareto front, with a third
    objective represented by the colour of each solution.

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
        color_obj
    ]

    unknown = set(selected_names).difference(objective_names)

    if unknown:
        raise ValueError(
            f"Unknown objective name(s): {sorted(unknown)}"
        )

    # The three objectives must be different
    if len(set(selected_names)) != 3:
        raise ValueError(
            "obj1, obj2 and color_obj "
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

    # Create figure
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # Plot Pareto front
    scatter = ax.scatter(
        x_values,
        y_values,
        c=colour_values,
        cmap="viridis_r",
        s=35,
        alpha=0.8
    )

    ax.set_xlabel(obj1)
    ax.set_ylabel(obj2)

    # Ideal point
    # All objectives in this optimisation problem are
    # minimised, so the ideal point is the minimum value
    # of each selected objective.

    ideal_x = data[obj1].min()
    ideal_y = data[obj2].min()

    ax.scatter(
        ideal_x,
        ideal_y,
        marker="X",
        s=100,
        color="black",
        linewidths=1.5,
        label="Ideal point"
    )

    # Build title using plot_info
    title = (
        f"Pareto front: {obj2} vs {obj1}"
        f" | colour = {color_obj}"
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

    # Legend
    ax.legend(
        loc="best",
        frameon=True
    )

    # Grid and layout
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    plt.show()

    return fig, ax

# Function to plot a Pareto front with four objectives, using position, colour, and marker size.
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
        title += f" – {scenario_name}"

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
        title += f" ({scenario_name})"

    ax.set_title(title)

    ax.grid(True)

    if duration_days is not None:
        ax.set_xlim(0, duration_days)

    fig.tight_layout()

    plt.show()

    return fig, ax