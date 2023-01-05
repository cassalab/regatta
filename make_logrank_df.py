import sys
import pandas as pd
import numpy as np
import copy
import scipy.stats
import itertools
import json
import pickle
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from config import acceptable_pathogenic_annotations, gene_to_end_position, src_path
sys.path.insert(1, src_path + "src/")
from jenks import load_clinvar_df, getJenksBreaks, altered_generate_coding_position_list
from utils import get_34_gene_list, get_using_genes
from survival_tools import annotate_region_diagram_ax

def generate_logrank_df():
    hr_diffs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    all_logrank_vals = []
    for hr_diff in hr_diffs:
        logrank_vals = []
        for gene_of_interest in mega_file.keys():
            plt.close("all")
            plt.close()
            vals = mega_file[gene_of_interest]
            fig, axes = plt.subplots(len(vals.keys()), 1, figsize = (30, len(vals.keys()) * 2))
            fig.tight_layout()
            fig.subplots_adjust(
                left=None,
                bottom=None,
                right=None,
                top=None,
                wspace=None,
                hspace=1.7
            )

            strand_fontsize = 40
            for i, (key, values) in enumerate(vals.items()):
                plt.close('all')
                try:
                    gene_df = gene_to_df[gene_of_interest]
                    gene_df = gene_df.loc[
                        gene_df["Deleterious"] == 0
                    ]
                    region_to_weight = {}
                    breakpoints = values["breakpoints"]
                    breakpoints[-1] = gene_to_end_position[gene_of_interest]
                    missense_weight = regression_results.loc[gene_of_interest]["Missense exp(coef)"]
                    high_regions = []
                    low_regions = []
                    for k2, v2 in values.items():
                        if type(k2) == int:
                            weight = v2['exp(coef)']
                            region_to_weight[k2] = weight
                            if weight / missense_weight >= (1 + hr_diff):
                                high_regions.append(k2)
                            elif weight / missense_weight <= (1 - hr_diff):
                                low_regions.append(k2)

                    color_path = "../data/pymol_color_scripts/" + gene_of_interest + "/" + gene_of_interest + "_" + str(len(region_to_weight))
                    annotate_region_diagram_ax(
                        axes[i],
                        breakpoints,
                        region_to_weight,
                        regression_results.loc[gene_of_interest]["Missense exp(coef)"],
                        color_path = color_path
                    )
                    axes[i].text(
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

                        axes[i].text(
                            np.max(breakpoints) + (np.max(breakpoints) * 0.02),
                            0.1,
                            p_string,
                            fontsize = 25
                        )

                        logrank_vals.append({
                            "key" : gene_of_interest + "_" + str(len(breakpoints) - 1) + "_" + str(hr_diff),
                            "p" : logrank_p,
                            "gene" : gene_of_interest,
                            "breaks" : len(breakpoints) - 1,
                            "hr_diff" : hr_diff,
                            "chisq_p" : values["chi_sq_p"],
                            "breakpoints" : "|".join(list(map(lambda x : str(x), breakpoints))),
                            "high_regions" : "|".join(list(map(lambda x : str(x), high_regions))),
                            "low_regions" : "|".join(list(map(lambda x : str(x), low_regions)))
                        })
                    else:
                        pass

                except Exception as e:
                    pass
            fig.suptitle("$\it{" + gene_of_interest + "}$", y = 1.05, fontsize = 40)
            fig.savefig(f"./scratch/region_diagrams_092922/{gene_of_interest}_{hr_diff}.png", bbox_inches = "tight")
        logrank_df = pd.DataFrame(logrank_vals)
        logrank_df = logrank_df.sort_values(by = "p")
        alpha = 0.01
        m = len(logrank_df)
        signif_genes = set()
        for counter, (index, row) in enumerate(logrank_df.iterrows()):
            if row["p"] <= ((counter + 1) / m) * alpha:
                if row["gene"] not in signif_genes:
                    print(row["gene"], row["p"], row["breaks"])
                signif_genes.add(row["gene"])
       all_logrank_vals = all_logrank_vals + logrank_vals

    all_logrank_df = pd.DataFrame(all_logrank_vals)
    all_logrank_df = all_logrank_df.set_index("key")

if __name__ == "__main__":
    global gene_to_df
    global regression_results
    global mega_file
    file = open("gene_to_df_delet_removed_092922.pickle",'rb')
    gene_to_df = pickle.load(file)
    regression_results = pd.read_csv("regression_results_delet_removed_092922.csv", index_col = 0)
    file = open("mega_results_checkppoint_092922.pickle",'rb')
    mega_file = pickle.load(file)
    generate_logrank_df()
