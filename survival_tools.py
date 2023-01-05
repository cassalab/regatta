import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import itertools
import lifelines
import glob
import os
import copy
from matplotlib import gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys
sys.path.insert(1,"/Users/jdf36/Desktop/bwh/cancer_regions/src/")
from jenks import load_clinvar_df, getJenksBreaks, altered_generate_coding_position_list



def annotate_region_diagram_ax(ax, jenks_regions, region_to_weight, missense_haz, dist_offset = 15,color_path = None):
    rotation = 45
    ax.set_xticks(jenks_regions)
    labels = []
    for item in jenks_regions:
        labels.append("c." + str(item))

    region_label_height = 0.45
    height = 1.0
    text_fontsize = 15
    title_fontsize = 23

    ax.set_xticklabels(labels, rotation = rotation, fontsize = text_fontsize, fontweight = "bold")
    counter = 1

    if color_path is not None:
        path = color_path.split("/")[:-1]
        path_string = "/".join(path)
        if not os.path.exists(path_string):
            os.makedirs(path_string)
        with open(color_path, "w") as color_file:
            cmap = LinearSegmentedColormap.from_list('mycmap', ['powderblue', 'white', 'salmon'])
            norm = Normalize(vmin=-1, vmax=1)
            for index, item in enumerate(jenks_regions):
                if index == len(jenks_regions) - 1:
                    pass
                else:
                    width = jenks_regions[index + 1] - item
                    weight = np.log2(region_to_weight[index + 1] / missense_haz)
                    color =  cmap(norm(np.log2(region_to_weight[index + 1] / missense_haz)))
                    ax.add_patch(patches.Rectangle((item, 0), width, height, color = color, ec="black", lw = 2, ls = "-"))
                    color_hex = matplotlib.colors.to_hex(color).replace("#", "0x")
                    to_write = "color " + color_hex + ", resi " + str(int(item / 3)) + "-"
                    color_file.write(to_write + "\n")
        color_file.close()

    else:
        cmap = LinearSegmentedColormap.from_list('mycmap', ['powderblue', 'white', 'salmon'])
        norm = Normalize(vmin=-1, vmax=1)
        for index, item in enumerate(jenks_regions):
            if index == len(jenks_regions) - 1:
                pass
            else:
                width = jenks_regions[index + 1] - item
                color =  cmap(norm(np.log2(region_to_weight[index + 1] / missense_haz)))
                if np.log2(region_to_weight[index + 1] / missense_haz) < -0.8:
                    color = cmap(norm(np.log2(1)))
                ax.add_patch(patches.Rectangle((item, 0), width, height, color = color, ec="black", lw = 2, ls = "-"))

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")

    prev_pos = None
    for i, bound in enumerate(jenks_regions):
        if prev_pos is None:
            prev_pos = bound
            continue

        dist = bound - ((bound - prev_pos) / 2)
        ax.text(dist - dist_offset, region_label_height, str(i), fontsize = text_fontsize, fontweight = "bold")
        prev_pos = bound

    ax.get_yaxis().set_visible(False)


def get_pfam_domains(gene):
    pfam_path = "/Users/jdf36/Desktop/bwh/cancer_regions//"
    f = open(pfam_path + gene + '.json')
    data = json.load(f)
    domain_to_regions = {}
    for entry in data["regions"]:
        name = entry["metadata"]["description"]
        start = entry["metadata"]["start"] * 3
        end = entry["metadata"]["end"] * 3
        if name in domain_to_regions.keys():
            domain_to_regions[name].append([start, end])
        else:
            domain_to_regions[name] = [[start, end]]
    f.close()
    return domain_to_regions


def get_single_gene_and_breaks(gene_of_interest, num_breaks, high_regions = [], low_regions = [], hr_diff = 0.25, annotate_pfam = False, existing_ax = None):
    domain_colors = ["seagreen", "brown", "navy", "violet", "cadetblue"]

    file = open("/Users/jdf36/Desktop/bwh/cancer_regions/src/mega_results_checkppoint_3.pickle",'rb')
    mega_file = pickle.load(file)
    vals = mega_file[gene_of_interest]
    gene_df = gene_to_df[gene_of_interest]
    gene_df = gene_df.loc[
        gene_df["Deleterious"] == 0
    ]
    if existing_ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (35, 2))
    else:
        ax = existing_ax
    strand_fontsize = 40
    region_to_weight = {}

    missense_weight = regression_results.loc[gene_of_interest]["Missense exp(coef)"]
    high_regions = []
    low_regions = []
    values = vals[num_breaks]
    breakpoints = values["breakpoints"]
    for k2, v2 in values.items():
        if type(k2) == int:
            weight = v2['exp(coef)']
            region_to_weight[k2] = weight
            if weight / missense_weight >= (1 + hr_diff):
                high_regions.append(k2)
            elif weight / missense_weight <= (1 - hr_diff):
                low_regions.append(k2)

    annotate_region_diagram_ax(
        ax,
        breakpoints,
        region_to_weight,
        regression_results.loc[gene_of_interest]["Missense exp(coef)"],
        color_path = color_path
    )

    ax.text(
        np.max(breakpoints) + (np.max(breakpoints) * 0.02),
        0.7,
        "$\chi^2 \ \ \ p \ = " + str(round(values["chi_sq_p"], 4)) + "$",
        fontsize = 25
    )

    if len(high_regions) > 0 and len(low_regions) > 0:
        T_high = []
        E_high = []
        for h_reg in high_regions:
            break_upper = breakpoints[h_reg]
            break_lower = breakpoints[h_reg - 1]
            carriers = gene_df.loc[
                (gene_df["coding_position"] >= break_lower) &
                (gene_df["coding_position"] < break_upper)
            ]
            T_high = T_high + list(carriers["breastcancer_age"].values)
            E_high = E_high + list(carriers["breastcancer"].values)

        T_low = []
        E_low = []
        for l_reg in low_regions:
            break_upper = breakpoints[l_reg]
            break_lower = breakpoints[l_reg - 1]
            carriers = gene_df.loc[
                (gene_df["coding_position"] >= break_lower) &
                (gene_df["coding_position"] < break_upper)
            ]
            T_low = T_low + list(carriers["breastcancer_age"].values)
            E_low = E_low + list(carriers["breastcancer"].values)

        logrank = logrank_test(
            T_high,
            T_low,
            event_observed_A=E_high,
            event_observed_B=E_low
        )
        logrank_p = logrank.p_value
        p_string = "$logrank \ p \ = " + str(round(logrank_p, 4)) + "$"
        if logrank_p < 0.0001:
            sci_notation_pieces = "{:.2e}".format(logrank_p).split("e")
            exponent = int(sci_notation_pieces[1])
            decimal = sci_notation_pieces[0]
            p_string = "$logrank \ p \ = " + decimal + " x 10^{" + str(exponent) + "}" +  "$"
            ax.text(
                np.max(breakpoints) + (np.max(breakpoints) * 0.02),
                0.3,
                p_string,
                fontsize = 25
            )
    fig.suptitle("$\it{" + gene_of_interest + "}$", y = 1.5, fontsize = 40)
    if annotate_pfam:
        patch_list = []
        pfam = get_pfam_domains(gene_of_interest)
        for index, (k, v) in enumerate(pfam.items()):
            for r in v:
                ax.add_patch(plt.Rectangle((r[0], -1.3), r[1] - r[0], 0.2, facecolor=domain_colors[index], clip_on=False,linewidth = 0))
                patch = patches.Patch(color=domain_colors[index], label=k)
            patch_list.append(patch)

        ax.legend(handles=patch_list, loc = "lower left", bbox_to_anchor=(0,-3.2), fontsize = 20)

        fig.tight_layout()



def risk_ratio(
    high_risk_group,
    low_risk_group,
    condition = "breastcancer",
    t = 65,
    return_confidence = False,
    z = 1.96
):
    ## CI calcuated by
    ## https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals8.html


    age_restricted_high = high_risk_group.loc[
        ((high_risk_group[condition] == 1) & (high_risk_group[f"{condition}_age"] < t)) |
        ((high_risk_group[condition] == 0) & (high_risk_group[f"{condition}_age"] >= t))
    ]
    age_restricted_low  = low_risk_group.loc[
        ((low_risk_group[condition] == 1) & (low_risk_group[f"{condition}_age"] < t)) |
        ((low_risk_group[condition] == 0) & (low_risk_group[f"{condition}_age"] >= t))
    ]
    IE =  len(age_restricted_high.loc[age_restricted_high[condition] == 1])
    IN = len(age_restricted_high.loc[age_restricted_high[condition] == 0])
    CE = len(age_restricted_low.loc[age_restricted_low[condition] == 1])
    CN = len(age_restricted_low.loc[age_restricted_low[condition] == 0])

    if (IE + IN) == 0 or (CE + CN) == 0 or IE == 0 or CE == 0:
        return None


    RR = (IE / (IE + IN)) / (CE / (CE + CN))

    t1 = (IN / IE) / (IE + IN)
    t2 = (CN / CE) / (CE + CN)
    CI_size = z * np.sqrt(t1 + t2)

    if return_confidence:
        return  round(RR, 2), round(RR - CI_size, 2), round(RR + CI_size,2), len(age_restricted_high), len(age_restricted_low), [[IE, IN], [CE, CN]]

    return round(RR, 2), len(age_restricted_high), len(age_restricted_low)

def get_km_curve_ax(
    ax,
    T1,
    T2,
    E1,
    E2,
    color_group_1 = "red",
    color_group_2 = "blue",
    label_group_1 = "",
    label_group_2 = "",
    title = "",
    use_ylabel = True,
    use_xlabel= True,
    show_at_risk_counts = True,
    show_logrank = True,
    at_risk_y_offset = -0.1,
    x_tick_plot = None,
    legend_fontsize = 15
):
    p_val_height = 0.97
    sns.set_style("whitegrid")
    kmf1 = KaplanMeierFitter()
    kmf1.fit(T1, E1)
    kmf1.plot(ax = ax, color = color_group_1, label = label_group_1)
    kmf2 = KaplanMeierFitter()
    kmf2.fit(T2, E2)
    kmf2.plot(ax = ax, color = color_group_2, label = label_group_2)
    x_ticks = list(np.arange(0,71,10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(list(np.arange(0,71,10)), fontsize = 15)
    y_ticks = [0.6,0.7,0.8,0.9,1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize = 15)
    ax.set_xlim(right = 70.01, left=0)
    if use_xlabel:
        ax.set_xlabel("Age", fontsize = 20)
    ax.set_title(title, fontsize = 25)
    if use_ylabel:
        ax.set_ylabel("Proportion Unaffected \n by Breast Cancer", fontsize = 20)
    else:
        ax.set_ylabel("")
    logrank = logrank_test(
        T1,
        T2,
        event_observed_A=E1,
        event_observed_B=E2
    )
    logrank_p = logrank.p_value
    p_string = "$logrank \ p \ = " + str(round(logrank_p, 4)) + "$"
    if logrank_p < 0.0001:
        sci_notation_pieces = "{:.2e}".format(logrank_p).split("e")
        exponent = int(sci_notation_pieces[1])
        decimal = sci_notation_pieces[0]
        p_string = "$logrank \ p \ = " + decimal + " x 10^{" + str(exponent) + "}" +  "$"
        p_val_height = 0.96

    ax.legend(loc = "lower left", fontsize = legend_fontsize)
    if show_logrank:
        ax.text(2.7, p_val_height, p_string, bbox={"color": "lightgray"}, fontsize = 17)

    if show_at_risk_counts:

        if x_tick_plot is None:
            x_tick_plot = copy.deepcopy(x_ticks)

        ax.set_xticks(x_tick_plot)
        ax.set_xlim(left = np.min(x_tick_plot))
        # ax.set_xlabel("AGEEEE \n\n\n\n")
        lifelines.plotting.add_at_risk_counts(
            kmf1,
            kmf2,
            labels = ["", ""],
            ax = ax,
            xticks = x_tick_plot,
            rows_to_show = ['At risk'],
            fontsize = 20,
            y = at_risk_y_offset
        )
        ax.set_xticks(x_tick_plot)
        ax.set_yticks(y_ticks)
    if use_xlabel:
        ax.set_xlabel("Age", fontsize = 20)
    else:
        ax.set_xlabel("", fontsize = 20)



def get_colors(gene, breaks):
    with open("/Users/jdf36/Desktop/bwh/cancer_regions/data/pymol_color_scripts/" + gene + "/" + gene + "_" + str(breaks), "r") as f:
        lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        c = line.split(",")[0].replace("color ", "").replace("0x", "#")
        colors.append(c)
    return colors

def get_pfam_domains(gene):
    pfam_path = "../../data/pfam_json/"
    f = open(pfam_path + gene + '.json')
    data = json.load(f)
    domain_to_regions = {}
    for entry in data["regions"]:
        name = entry["metadata"]["description"]
        start = entry["metadata"]["start"] * 3
        end = entry["metadata"]["end"] * 3
        if name in domain_to_regions.keys():
            domain_to_regions[name].append([start, end])
        else:
            domain_to_regions[name] = [[start, end]]
    f.close()
    return domain_to_regions


def position_in_pfam_lists(pos, lists):
    if type(pos) not in [float, int]:
        return 0
    for lst in lists:
        if pos >= lst[0] and pos <= lst[1]:
            return 1
    return 0



def get_annotated_pfam_profiles(gene, use_preload = True, df = None):
    if use_preload:
        gene_df = copy.deepcopy(gene_to_df[gene])
    else:
        gene_df = copy.deepcopy(df)
    pfam_domains = get_pfam_domains(gene)
    valid_people = gene_df.loc[
        (gene_df["Deleterious"] == 0)
    ]
    valid_people = gene_df
    for domain, region_list in pfam_domains.items():
        valid_people[domain] = valid_people["coding_position"].apply(lambda x : position_in_pfam_lists(x, region_list))

    all_pfam_domains = list(itertools.chain.from_iterable(pfam_domains.values()))
    valid_people["any domain"] = valid_people["coding_position"].apply(lambda x : position_in_pfam_lists(x, all_pfam_domains))
    return valid_people


def get_logrank_p(T1, T2, E1, E2):
    logrank = logrank_test(
        T1,
        T2,
        event_observed_A=E1,
        event_observed_B=E2
    )

    logrank_p = logrank.p_value
    return logrank_p




def get_annotated_hr_lr_regions(
    gene,
    breakpoints,
    hr_regions,
    lr_regions,
    use_preload = True,
    gene_to_df_loc = "/Users/jdf36/Desktop/bwh/cancer_regions/src/gene_to_df_delet_removed_030821.pickle",
    df = None
):
    if use_preload:
        file = open(gene_to_df_loc, 'rb')
        gene_to_df = pickle.load(file)
        gene_df = copy.deepcopy(gene_to_df[gene])
    else:
        gene_df = copy.deepcopy(df)

    gene_df = gene_df.loc[gene_df["Deleterious"]==0]

    hr_carriers = set()
    for h_region in hr_regions:
        start = breakpoints[h_region - 1]
        end = breakpoints[h_region]
        region_carriers = gene_df.loc[
            (gene_df["coding_position"] >= start) &
            (gene_df["coding_position"] < end)
        ]
        hr_carriers = hr_carriers.union(region_carriers.index)

    lr_carriers = set()
    for l_region in lr_regions:
        start = breakpoints[l_region - 1]
        end = breakpoints[l_region]
        region_carriers = gene_df.loc[
            (gene_df["coding_position"] >= start) &
            (gene_df["coding_position"] < end)
        ]
        lr_carriers = lr_carriers.union(region_carriers.index)

    gene_df["low_region_carrier"] = [1 if x in lr_carriers else 0 for x in list(gene_df.index)]
    gene_df["high_region_carrier"] = [1 if x in hr_carriers else 0 for x in list(gene_df.index)]
    return gene_df

def get_pfam_comparison_logrank_vals(gene):
    pfam_profiles = get_annotated_pfam_profiles(gene)
    pfam_carriers = pfam_profiles.loc[
        (pfam_profiles["any domain"] == 1) &
        (pfam_profiles["Missense"] == 1)
    ]
    non_pfam_carriers = pfam_profiles.loc[
        (pfam_profiles["any domain"] == 0) &
        (pfam_profiles["Missense"] == 1)
    ]
    non_carriers = pfam_profiles.loc[
        (pfam_profiles["Missense"] == 0)
    ]
    E1 = pfam_carriers["breastcancer"].values
    T1 = pfam_carriers["breastcancer_age"].values


    E2 = non_pfam_carriers["breastcancer"].values
    T2 = non_pfam_carriers["breastcancer_age"].values

    E3 = non_carriers["breastcancer"].values
    T3 = non_carriers["breastcancer_age"].values
    d = {
        "pfam_vs_non_pfam_missense" : get_logrank_p(T1, T2, E1, E2),
        "pfam_vs_non_carrier" : get_logrank_p(T1, T3, E1, E3)
    }
    return d




def make_hr_lr_binary_diagram(
    gene_of_interest,
    breaks,
    hr_regions,
    lr_regions,
    colors,
    title = "",
    color_high = "salmon",
    color_low = "powderblue",
    annotate_pfam = True,
    dist_offset = 15,
    ax = None,
    text_fontsize = 25,
    title_fontsize = 23,
    use_hatches = True,
    label_regions = True
):
    domain_colors = ["seagreen", "brown", "navy", "violet", "cadetblue", "crimson", "sienna", "burlywood"]
    sns.set_style("white")
    region_label_height = 0.45
    height = 1.0
    rotation = 45
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (35, 2))
    ax.set_xticks(breaks)
    labels = []
    for item in breaks:
        labels.append("c." + str(int(item)))

    ax.set_xticklabels(labels, rotation = rotation, fontsize = text_fontsize, fontweight = "bold")
    counter = 1
    height = 1
    for index, item in enumerate(breaks):

        if index == len(breaks) - 1:
            pass
        else:
            color = colors[index]
            width = breaks[index + 1] - item
            if ((index + 1 in hr_regions) or (index + 1 in lr_regions)) and use_hatches:
                ax.add_patch(patches.Rectangle(
                    (item, 0),
                        width,
                        height,
                        color = color,
                        ec="black",
                        lw = 2,
                        ls = "-",
                        hatch=r"/"
                    )
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (item, 0),
                        width,
                        height,
                        color = color,
                        ec="black",
                        lw = 2,
                        ls = "-")
                )

    if annotate_pfam:
        patch_list = []
        pfam = get_pfam_domains(gene_of_interest)
        for index, (k, v) in enumerate(pfam.items()):
            for r in v:
                ax.add_patch(plt.Rectangle((r[0], -1.3), r[1] - r[0], 0.2, facecolor=domain_colors[index], clip_on=False,linewidth = 0))
                patch = patches.Patch(color=domain_colors[index], label=k)
            patch_list.append(patch)

        ax.legend(handles=patch_list, loc = "lower left", bbox_to_anchor=(0,-3.2), fontsize = 20)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")

    prev_pos = None
    for i, bound in enumerate(breaks):
        if prev_pos is None:
            prev_pos = bound
            continue

        color = colors[i - 1]

        dist = bound - ((bound - prev_pos) / 2)
        if label_regions == True:
            t = ax.text(dist - (dist_offset if i < 11 else dist_offset * 1.5), region_label_height, str(i), fontsize = text_fontsize, fontweight = "bold")
            t.set_bbox(dict(facecolor=color, edgecolor=color))
        prev_pos = bound

    ax.get_yaxis().set_visible(False)
    ax.set_title("$\it{"+gene_of_interest+"}$\n", fontsize = 35)
    return ax
