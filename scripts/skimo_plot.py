from collections import namedtuple, defaultdict
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.collections as mcoll

import wandb


matplotlib.rcParams["pdf.fonttype"] = 42  # Important!!! Remove Type 3 fonts


def save_fig(file_name, file_format="pdf", tight=True, **kwargs):
    if tight:
        plt.tight_layout()
    file_name = "{}.{}".format(file_name, file_format).replace(" ", "-")
    plt.savefig(file_name, format=file_format, dpi=1000, **kwargs)


def export_legend(ax, line_width, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.axis("off")
    legend = ax2.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        loc="lower center",
        ncol=10,
        handlelength=2,
    )
    for line in legend.get_lines():
        line.set_linewidth(line_width)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def draw_line(
    log,
    method,
    avg_step=3,
    mean_std=False,
    max_step=None,
    max_y=None,
    x_scale=1.0,
    ax=None,
    color="C0",
    smooth_steps=10,
    num_points=50,
    line_style="-",
    marker=None,
    no_fill=False,
    smoothing_weight=0.0,
):
    steps = {}
    values = {}
    max_step = max_step * x_scale
    seeds = log.keys()
    is_line = True

    for seed in seeds:
        step = np.array(log[seed].steps)
        value = np.array(log[seed].values)

        if not np.isscalar(log[seed].values):
            is_line = False

            # filter NaNs
            for i in range(len(value)):
                if np.isnan(value[i]):
                    value[i] = 0 if i == 0 else value[i - 1]

        if max_step:
            max_step = min(max_step, step[-1])
        else:
            max_step = step[-1]

        steps[seed] = step
        values[seed] = value

    if is_line:
        y_data = [values[seed] for seed in seeds]
        std_y = np.std(y_data)
        avg_y = np.mean(y_data)
        min_y = np.min(y_data)
        max_y = np.max(y_data)

        l = ax.axhline(
            y=avg_y, label=method, color=color, linestyle=line_style, marker=marker
        )
        ax.axhspan(
            avg_y - std_y,  # max(avg_y - std_y, min_y),
            avg_y + std_y,  # min(avg_y + std_y, max_y),
            color=color,
            alpha=0.1,
        )
        return l, min_y, max_y

    # exponential moving average smoothing
    for seed in seeds:
        last = values[seed][:10].mean()  # First value in the plot (first timestep)
        smoothed = list()
        for point in values[seed]:
            smoothed_val = (
                last * smoothing_weight + (1 - smoothing_weight) * point
            )  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        values[seed] = smoothed

    # cap all sequences to max number of steps
    data = []
    for seed in seeds:
        for i in range(len(steps[seed])):
            if steps[seed][i] <= max_step:
                data.append((steps[seed][i], values[seed][i]))
    data.sort()
    x_data = []
    y_data = []
    for step, value in data:
        x_data.append(step)
        y_data.append(value)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    min_y = np.min(y_data)
    max_y = np.max(y_data)
    # l = sns.lineplot(x=x_data, y=y_data)
    # return l, min_y, max_y

    # filling
    if not no_fill:
        n = len(x_data)
        avg_step = int(n // num_points)

        x_data = x_data[: n // avg_step * avg_step].reshape(-1, avg_step)
        y_data = y_data[: n // avg_step * avg_step].reshape(-1, avg_step)

        std_y = np.std(y_data, axis=1)

        avg_x, avg_y = np.mean(x_data, axis=1), np.mean(y_data, axis=1)
    else:
        avg_x, avg_y = x_data, y_data

    # subsampling smoothing
    n = len(avg_x)
    ns = smooth_steps
    # avg_x = avg_x[: n // ns * ns].reshape(-1, ns).mean(axis=1)
    # avg_y = avg_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)
    # if not no_fill:
    #     std_y = std_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)

    if not no_fill:
        ax.fill_between(
            avg_x,
            avg_y - std_y,  # np.clip(avg_y - std_y, 0, max_y),
            avg_y + std_y,  # np.clip(avg_y + std_y, 0, max_y),
            alpha=0.1,
            color=color,
        )

    # horizontal line
    # if "SAC" in method:
    #     l = ax.axhline(
    #         y=avg_y[-1], xmin=0.1, xmax=1.0, color=color, linestyle="--", marker=marker
    #     )
    #     plt.setp(l, linewidth=2, color=color, linestyle="--", marker=marker)

    l = ax.plot(avg_x, avg_y, label=method)
    plt.setp(l, linewidth=2, color=color, linestyle=line_style, marker=marker)
    # 4 if 'Ours' not in method else 2

    return l, min_y, max_y


def draw_graph(
    plot_logs,
    line_logs,
    method_names=None,
    title=None,
    xlabel="Step",
    ylabel="Success",
    legend=False,
    mean_std=False,
    min_step=0,
    max_step=None,
    min_y=None,
    max_y=None,
    num_y_tick=5,
    smooth_steps=10,
    num_points=50,
    no_fill=False,
    num_x_tick=5,
    legend_loc=2,
    markers=None,
    smoothing_weight=0.0,
    file_name=None,
    line_styles=None,
    line_colors=None,
):
    if legend:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
    max_value = -np.inf
    min_value = np.inf

    if method_names is None:
        method_names = list(plot_logs.keys()) + list(line_logs.keys())

    lines = []
    num_colors = len(method_names)
    two_lines_per_method = False
    if "Pick" in method_names[0] or "Attach" in method_names[0]:
        two_lines_per_method = True
        num_colors = len(method_names) / 2

    for idx, method_name in enumerate(method_names):
        if method_name in plot_logs.keys():
            log = plot_logs[method_name]
        else:
            log = line_logs[method_name]

        seeds = log.keys()
        if len(seeds) == 0:
            continue

        color = (
            line_colors[method_name] if line_colors else "C%d" % (num_colors - idx - 1)
        )
        line_style = line_styles[method_name] if line_styles else "-"

        l_, min_, max_ = draw_line(
            log,
            method_name,
            mean_std=mean_std,
            max_step=max_step,
            max_y=max_y,
            x_scale=1.0,
            ax=ax,
            color=color,
            smooth_steps=smooth_steps,
            num_points=num_points,
            line_style=line_style,
            no_fill=no_fill,
            smoothing_weight=smoothing_weight[idx]
            if isinstance(smoothing_weight, list)
            else smoothing_weight,
            marker=markers[idx] if isinstance(markers, list) else markers,
        )
        # lines += l_
        max_value = max(max_value, max_)
        min_value = min(min_value, min_)

    if min_y == None:
        min_y = int(min_value - 1)
    if max_y == None:
        max_y = max_value
        # max_y = int(max_value + 1)

    # y-axis tick (belows are commonly used settings)
    if max_y == 1:
        plt.yticks(np.arange(min_y, max_y + 0.1, 0.2), fontsize=12)
    else:
        if max_y > 1:
            plt.yticks(
                np.arange(min_y, max_y + 0.01, (max_y - min_y) / num_y_tick),
                fontsize=12,
            )  # make this 4 for kitchen
        elif max_y > 0.8:
            plt.yticks(np.arange(0, 1.0, 0.2), fontsize=12)
        elif max_y > 0.5:
            plt.yticks(np.arange(0, 0.8, 0.2), fontsize=12)
        elif max_y > 0.3:
            plt.yticks(np.arange(0, 0.5, 0.1), fontsize=12)
        elif max_y > 0.2:
            plt.yticks(np.arange(0, 0.4, 0.1), fontsize=12)
        else:
            plt.yticks(np.arange(0, 0.2, 0.05), fontsize=12)

    # x-axis tick
    plt.xticks(
        np.round(
            np.arange(min_step, max_step + 0.1, (max_step - min_step) / num_x_tick), 2
        ),
        fontsize=12,
    )

    # background grid
    ax.grid(visible=True, which="major", color="lightgray", linestyle="--")

    # axis titles
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # set axis range
    ax.set_xlim(min_step, max_step)
    ax.set_ylim(bottom=-0.01, top=max_y + 0.01)  # use -0.01 to print ytick 0

    # print legend
    if legend:
        if isinstance(legend_loc, tuple):
            print("print legend outside of frame")
            leg = plt.legend(fontsize=15, bbox_to_anchor=legend_loc, ncol=6)
        else:
            leg = plt.legend(fontsize=15, loc=legend_loc)

    # export_legend(ax, 2, "learning_curve_legend.pdf")
    # for line in leg.get_lines():
    #     line.set_linewidth(2)
    # labs = [l.get_label() for l in lines]
    # plt.legend(lines, labs, fontsize='small', loc=2)

    # print title
    if title:
        plt.title(title, y=1.00, fontsize=16)

    # save plot to file
    if file_name:
        save_fig(file_name)


def get_data(description, x_scale=1000000, y_scale=1):
    api = wandb.Api()
    wandb_entity = "clvr"
    wandb_project = "p-skill-model"
    reports = api.reports(path=f"{wandb_entity}/{wandb_project}")
    for report in reports:
        print(report.description)
        if description == report.description:
            print("find", report.description)
            break
    else:
        raise ValueError("Could not find report")

    for spec in report.spec["blocks"]:
        if "metadata" in spec and "runSets" in spec["metadata"]:
            run_sets = spec["metadata"]["runSets"]
            break
    else:
        raise ValueError("Could not find run sets in report")

    Log = namedtuple("Log", ["values", "steps"])
    plot_logs = defaultdict(dict)
    op = None

    print("Run sets:")
    for run_set in run_sets:
        run_set_name = run_set["name"]
        if not run_set["enabled"]:
            print(f"  {run_set_name}  --  skipped")
            continue
        print(f"  {run_set_name}")
        runs = run_set["selections"]["tree"]

        for i, run_id in enumerate(runs):
            run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
            # df = run.history(samples=1000000)
            # df = df[["_step", "test_ep/rew"]].dropna()
            data = run.history(samples=10000)
            values = data["train_ep/rew"]
            if isinstance(y_scale, dict) and run_set_name in y_scale:
                values = values / y_scale[run_set_name]
            elif isinstance(y_scale, (int, float)):
                values = values / y_scale
            steps = data["_step"] / x_scale
            if op == "max":
                values = max(values)
            plot_logs[run_set_name][i] = Log(values, steps)
            print(f"    {i}: {run_id} ({len(steps)})")
    return plot_logs


def plot_maze():
    plot_logs = get_data("Final-Maze", y_scale=100)
    line_colors = {
        "Dreamer": "C5",
        "SPiRL": "C4",
        "SPiRL+Dreamer": "C3",
        "SkiMo+SAC": "C1",
        "SkiMo w/o joint training": "C0",
        "SkiMo (Ours)": "C2",
        "TD-MPC": "C9",
        "SPiRL+TDMPC": "C7",
        "SkiMo w/o CEM": "C5",
    }
    line_styles = {
        "Dreamer": "--",
        "SPiRL": ":",
        "SPiRL+Dreamer": "-.",
        "SkiMo+SAC": "-",
        "SkiMo w/o joint training": "-",
        "SkiMo (Ours)": "-",
        "TD-MPC": "--",
        "SPiRL+TDMPC": "-.",
        "SkiMo w/o CEM": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Success",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=2,
        min_y=0,
        max_y=1,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="learning_curve_maze",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_kitchen():
    plot_logs = get_data("Final-Kitchen")
    line_colors = {
        "Dreamer": "C5",
        "SPiRL": "C4",
        "SPiRL+Dreamer": "C3",
        "SkiMo+SAC": "C1",
        "SkiMo w/o joint training": "C0",
        "SkiMo (Ours)": "C2",
        "TD-MPC": "C9",
        "SPiRL+TDMPC": "C7",
        "SkiMo w/o CEM": "C5",
    }
    line_styles = {
        "Dreamer": "--",
        "SPiRL": ":",
        "SPiRL+Dreamer": "-.",
        "SkiMo+SAC": "-",
        "SkiMo w/o joint training": "-",
        "SkiMo (Ours)": "-",
        "TD-MPC": "--",
        "SPiRL+TDMPC": "-.",
        "SkiMo w/o CEM": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=True,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=1,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="learning_curve_kitchen",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_misaligned_kitchen():
    plot_logs = get_data("Final-Kitchen-Misaligned")
    line_colors = {
        "Dreamer": "C5",
        "SPiRL": "C4",
        "SPiRL+Dreamer": "C3",
        "SkiMo+SAC": "C1",
        "SkiMo w/o joint training": "C0",
        "SkiMo (Ours)": "C2",
        "TD-MPC": "C9",
        "SPiRL+TDMPC": "C7",
        "SkiMo w/o CEM": "C5",
    }
    line_styles = {
        "Dreamer": "--",
        "SPiRL": ":",
        "SPiRL+Dreamer": "-.",
        "SkiMo+SAC": "-",
        "SkiMo w/o joint training": "-",
        "SkiMo (Ours)": "-",
        "TD-MPC": "--",
        "SPiRL+TDMPC": "-.",
        "SkiMo w/o CEM": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=1,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="learning_curve_kitchen_misaligned",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_calvin():
    y_scale = {
        "Dreamer": 100,
        "SPiRL": 100,
        "SPiRL+Dreamer": 100,
        "TD-MPC": 100,
        "SPiRL+TDMPC": 100,
    }
    plot_logs = get_data("Final-Calvin", y_scale=y_scale)
    line_colors = {
        "Dreamer": "C5",
        "SPiRL": "C4",
        "SPiRL+Dreamer": "C3",
        "SkiMo+SAC": "C1",
        "SkiMo w/o joint training": "C0",
        "SkiMo (Ours)": "C2",
        "TD-MPC": "C9",
        "SPiRL+TDMPC": "C7",
        "SkiMo w/o CEM": "C5",
    }
    line_styles = {
        "Dreamer": "--",
        "SPiRL": ":",
        "SPiRL+Dreamer": "-.",
        "SkiMo+SAC": "-",
        "SkiMo w/o joint training": "-",
        "SkiMo (Ours)": "-",
        "TD-MPC": "--",
        "SPiRL+TDMPC": "-.",
        "SkiMo w/o CEM": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=2,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="learning_curve_calvin",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_ablation_kitchen(key=None):
    if key == "skill_horizon":
        val = ["Skill_Horizon", "horizon"]
    elif key == "planning_horizon":
        val = ["Planning_Horizon", "n_skill"]
    plot_logs = get_data(f"Ablations-Kitchen-{val[0]}")
    line_colors = {
        f"{val[1]}=1": "C5",
        f"{val[1]}=5": "C3",
        f"{val[1]}=10": "C1",
        f"{val[1]}=15": "C0",
        f"{val[1]}=20": "C2",
    }
    line_styles = {
        f"{val[1]}=1": "-",
        f"{val[1]}=5": "-",
        f"{val[1]}=10": "-",
        f"{val[1]}=15": "-",
        f"{val[1]}=20": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=1,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name=f"{key}_kitchen",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_ablation_maze(key=None):
    if key == "skill_horizon":
        val = ["Skill_Horizon", "H"]
    elif key == "planning_horizon":
        val = ["Planning_Horizon", "N"]
    plot_logs = get_data(f"Ablations-Maze-{val[0]}", y_scale=100)
    line_colors = {
        f"{val[1]}=1": "C5",
        f"{val[1]}=5": "C3",
        f"{val[1]}=10": "C1",
        f"{val[1]}=15": "C0",
        f"{val[1]}=20": "C2",
    }
    line_styles = {
        f"{val[1]}=1": "-",
        f"{val[1]}=5": "-",
        f"{val[1]}=10": "-",
        f"{val[1]}=15": "-",
        f"{val[1]}=20": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Success",  # y-axis title
        legend=True,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=2,
        min_y=0,
        max_y=1,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name=f"{key}_maze",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_cem_kitchen():
    plot_logs = get_data(f"Ablations-Kitchen-CEM")
    line_colors = {
        "cem": "C1",
        "no_cem": "C0",
    }
    line_styles = {
        "cem": "-",
        "no_cem": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=1,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name=f"cem_kitchen",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_cem_maze():
    plot_logs = get_data(f"Ablations-Maze-CEM", y_scale=100)
    line_colors = {
        "cem": "C1",
        "no_cem": "C0",
    }
    line_styles = {
        "cem": "-",
        "no_cem": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Success",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=2,
        min_y=0,
        max_y=1,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="cem_maze",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_cem_misaligned_kitchen():
    plot_logs = get_data("Final-Kitchen-Misaligned")
    line_colors = {
        "cem": "C1",
        "no_cem": "C0",
    }
    line_styles = {
        "cem": "-",
        "no_cem": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=1,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="cem_misaligned_kitchen",
        line_colors=line_colors,
        line_styles=line_styles,
    )


def plot_cem_calvin():
    plot_logs = get_data("Ablations-Calvin")
    line_colors = {
        "cem": "C1",
        "no_cem": "C0",
    }
    line_styles = {
        "cem": "-",
        "no_cem": "-",
    }
    draw_graph(
        plot_logs,  # curved lines
        {},  # line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel="Environment steps (1M)",  # x-axis title
        ylabel="Average Subtasks",  # y-axis title
        legend=False,
        legend_loc="lower right",  # (0.5, 1.2),
        max_step=2,
        min_y=0,
        max_y=4,
        num_y_tick=4,
        smooth_steps=1,
        num_points=100,
        num_x_tick=4,
        smoothing_weight=0.99,
        file_name="cem_calvin",
        line_colors=line_colors,
        line_styles=line_styles,
    )


if __name__ == "__main__":
    # Main experiments
    plot_maze()
    plot_kitchen()
    plot_misaligned_kitchen()
    plot_calvin()

    # Ablation study
    plot_ablation_maze(key="skill_horizon")
    plot_ablation_kitchen(key="skill_horizon")
    plot_ablation_maze(key="planning_horizon")
    plot_ablation_kitchen(key="planning_horizon")

    # plot_cem_maze()
    # plot_cem_kitchen()
    # plot_cem_calvin()
    # plot_cem_misaligned_kitchen()
