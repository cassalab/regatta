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
from config import acceptable_pathogenic_annotations, src_path
sys.path.insert(1, src_path + "src/")
from jenks import load_clinvar_df, getJenksBreaks, altered_generate_coding_position_list
from utils import get_34_gene_list, get_using_genes


def has_missense(v):
    if pd.isnull(v):
        return 0
    else:
        return 1

def get_all_delet_carriers(excluded_gene = None):
    lines = get_using_genes()

    if excluded_gene is not None:
        if excluded_gene in lines:
            lines.remove(excluded_gene)

    gene_to_df = pickle.load(open(src_path + "scratch/gene_to_df.pickle", "rb"))
    all_delet_carriers = set()
    for gene in lines:
        gene_df = copy.deepcopy(gene_to_df[gene])
        delet_carriers = gene_df.loc[
            gene_df["Deleterious"] == 1
        ]
        all_delet_carriers = all_delet_carriers.union(
            set(delet_carriers.index.values)
        )

    return all_delet_carriers


def get_sequenced():
    with open(src_path + "data/sequenced_200k.txt", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x : int(x),lines))
    f.close()
    return lines


def place_position(pos_list, pos):
    i = 0
    while pos > pos_list[i] and i < len(pos_list) - 1:
        i += 1
    return i - 1


def make_gene_to_df():
    sequenced_people = set(get_sequenced())
    gene_to_df = {}
    results_dicts = []
    genes = get_using_genes()
    for g in genes:
        carriers = pd.read_csv(f"{src_path}data/carriers/{g}.csv")
        variant_info = pd.read_csv(f"{src_path}data/variant_files/{g}.csv")
        patient_to_phenotype = pd.read_csv(f"{src_path}data/updated_patient_to_phenotype_041721.csv", index_col =0)
        variant_to_carriers = dict(zip(carriers["variant"], carriers["patient"]))
        missense_variant_df = variant_info.loc[
            variant_info["consequence"] == "Missense"
        ]
        delet_varaint_df = variant_info.loc[
            variant_info["vep_consequence"].str.contains("frameshift")  |
            variant_info["vep_consequence"].str.contains("stop_gained") |
            variant_info["vep_consequence"].str.contains("start_lost") |
            variant_info["vep_consequence"].str.contains("start_lost") |
            variant_info["vep_consequence"].str.contains("splice_donor") |
            variant_info["vep_consequence"].str.contains("splice_acceptor")
        ]
        missense_variants = set(missense_variant_df["Name"].values)
        delet_variants = set(delet_varaint_df["Name"].values)
        delet_carriers = set()
        for variant in delet_variants:
            if variant in variant_to_carriers.keys():
                c = variant_to_carriers[variant]
                if type(c) == float:
                    continue
                c = list(map(lambda x : int(x), c.split("|")))
                for participant in c:
                    delet_carriers.add(participant)

        patient_to_variants = {}
        for v, p_list in variant_to_carriers.items():
            if v not in missense_variants:
                continue
            if type(p_list) == float:
                continue
            carriers = list(map(lambda x : int(x), p_list.split("|")))
            if len(carriers) >= 0.005*200000:
                continue
            for c in carriers:
                if c in patient_to_variants.keys():
                    patient_to_variants[c].append(v)
                else:
                    patient_to_variants[c] = [v]

        patient_to_variant = {}
        patients_to_remove = set()
        variant_to_codong_position = dict(
            zip(
                missense_variant_df["Name"].values,
                missense_variant_df["coding_position"].values
            )
        )
        for p, variants in patient_to_variants.items():
            if len(variants) == 1:
                patient_to_variant[p] = [variants[0], variant_to_codong_position[variants[0]]]
            else:
                # carry multiple missense variants
                patients_to_remove.add(p)

        to_remove = patients_to_remove.union(get_all_delet_carriers(excluded_gene = g))
        df = pd.DataFrame.from_dict(
            patient_to_variant,
            orient = "index",
            columns = ["variant", "coding_position"]
        )

        regression_df = patient_to_phenotype.join(df)
        regression_df["Missense"] = regression_df["variant"].apply(lambda x : has_missense(x))
        regression_df["Deleterious"] = [1 if x in delet_carriers else 0 for x in list(regression_df.index)]
        regression_df["to_drop"] = [1 if x in to_remove else 0 for x in list(regression_df.index)]
        regression_df["sequenced"] = [1 if x in sequenced_people else 0 for x in list(regression_df.index)]
        regression_df = regression_df.loc[
            (regression_df["to_drop"] == 0) &
            (regression_df["sex_int"] == 0) &
            (regression_df["sequenced"] == 1)
        ]

        gene_to_df[g] = regression_df

    filehandler = open("gene_to_df_delet_removed_09222.pickle","wb")
    pickle.dump(gene_to_df, filehandler)

def get_regressions_results():
    sequenced_people = set(get_sequenced())
    gene_to_df = {}
    results_dicts = []
    genes = get_using_genes()
    for g in genes:
        carriers = pd.read_csv(f"{src_path}data/carriers/{g}.csv")
        variant_info = pd.read_csv(f"{src_path}data/variant_files/{g}.csv")
        patient_to_phenotype = pd.read_csv(f"{src_path}data/updated_patient_to_phenotype_041721.csv", index_col =0 )
        variant_to_carriers = dict(zip(carriers["variant"], carriers["patient"]))
        missense_variant_df = variant_info.loc[
            variant_info["consequence"] == "Missense"
        ]

        delet_varaint_df = variant_info.loc[
            variant_info["vep_consequence"].str.contains("frameshift")  |
            variant_info["vep_consequence"].str.contains("stop_gained") |
            variant_info["vep_consequence"].str.contains("start_lost") |
            variant_info["vep_consequence"].str.contains("start_lost") |
            variant_info["vep_consequence"].str.contains("splice_donor") |
            variant_info["vep_consequence"].str.contains("splice_acceptor")
        ]

        missense_variants = set(missense_variant_df["Name"].values)
        delet_variants = set(delet_varaint_df["Name"].values)
        delet_carriers = set()
        for variant in delet_variants:
            if variant in variant_to_carriers.keys():
                c = variant_to_carriers[variant]
                if type(c) == float:
                    continue
                c = list(map(lambda x : int(x), c.split("|")))
                for participant in c:
                    delet_carriers.add(participant)

        patient_to_variants = {}
        for v, p_list in variant_to_carriers.items():
            if v not in missense_variants:
                continue
            if type(p_list) == float:
                continue
            carriers = list(map(lambda x : int(x), p_list.split("|")))
            if len(carriers) >= 0.005*200000:
                continue
            for c in carriers:
                if c in patient_to_variants.keys():
                    patient_to_variants[c].append(v)
                else:
                    patient_to_variants[c] = [v]

        patient_to_variant = {}
        patients_to_remove = set()
        variant_to_codong_position = dict(
            zip(
                missense_variant_df["Name"].values,
                missense_variant_df["coding_position"].values
            )
        )
        for p, variants in patient_to_variants.items():
            if len(variants) == 1:
                patient_to_variant[p] = [variants[0], variant_to_codong_position[variants[0]]]
            else:
                # carry multiple missense variants
                patients_to_remove.add(p)

        to_remove = patients_to_remove.union(get_all_delet_carriers(excluded_gene = g))

        df = pd.DataFrame.from_dict(
            patient_to_variant,
            orient = "index",
            columns = ["variant", "coding_position"]
        )

        regression_df = patient_to_phenotype.join(df)
        regression_df["Missense"] = regression_df["variant"].apply(lambda x : has_missense(x))
        regression_df["Deleterious"] = [1 if x in delet_carriers else 0 for x in list(regression_df.index)]
        regression_df["to_drop"] = [1 if x in to_remove else 0 for x in list(regression_df.index)]
        regression_df["sequenced"] = [1 if x in sequenced_people else 0 for x in list(regression_df.index)]
        regression_df = regression_df.loc[
            (regression_df["to_drop"] == 0) &
            (regression_df["sex_int"] == 0) &
            (regression_df["sequenced"] == 1)
        ]
        gene_to_df[g] = regression_df
        all_results = {}
        regression_df_del = copy.deepcopy(regression_df)
        regression_df_del = regression_df_del.loc[regression_df_del["Missense"] == 0]
        cols_keeping = ["Deleterious", "breastcancer_age", "breastcancer"]
        cols = copy.deepcopy(list(regression_df_del.columns))
        for c in cols_keeping:
            cols.remove(c)

        regression_df_del = regression_df_del.drop(columns = cols)
        fitter = CoxPHFitter()
        fitter.fit(regression_df_del, event_col ="breastcancer", duration_col = "breastcancer_age")
        res = fitter.summary.loc["Deleterious"].to_dict()
        res["carriers"] = len(regression_df_del.loc[regression_df_del["Deleterious"] == 1])

        for k, v in res.items():
            all_results["Deleterious " + k] = v

        regression_df_mis = copy.deepcopy(regression_df)
        regression_df_mis = regression_df_mis.loc[regression_df_mis["Deleterious"] == 0]
        cols_keeping = ["Missense", "breastcancer_age", "breastcancer"]
        cols = copy.deepcopy(list(regression_df_mis.columns))
        for c in cols_keeping:
            cols.remove(c)
        regression_df_mis = regression_df_mis.drop(columns = cols)

        fitter = CoxPHFitter()
        fitter.fit(regression_df_mis, event_col ="breastcancer", duration_col = "breastcancer_age")
        res = fitter.summary.loc["Missense"].to_dict()


        res["carriers"] = len(regression_df_mis.loc[regression_df_mis["Missense"] == 1])
        res["concordance"] = fitter.concordance_index_
        for k, v in res.items():
            all_results["Missense " + k] = v

        all_results["gene"] = g
        results_dicts.append(all_results)

    filehandler = open("gene_to_df_delet_removed_092922.pickle","wb")
    pickle.dump(gene_to_df, filehandler)

    return results_dicts

def make_metrics_file():
    bc_non_limited = ["MSH6"]
    results_of_het_test = []
    gene_to_regions_results_df =[]
    genes_of_interest = list(results.index)
    results_mega_dict = {}
    gene_to_df = pickle.load(open("gene_to_df_delet_removed_092922.pickle", "rb"))
    for gene_of_interest in genes_of_interest:
        results_mega_dict[gene_of_interest] = {}
        min_p = float("inf")
        argmin_p = None
        baseline = results.loc[gene_of_interest][["Missense coef", "Missense exp(coef)", "Missense p"]]
        for n_breaks in range(2, 16):
            results_mega_dict[gene_of_interest][n_breaks] = {}
            try:
                gene_df = copy.deepcopy(gene_to_df[gene_of_interest])

                limit_breast_cancer = True
                if gene_of_interest in bc_non_limited:
                    limit_breast_cancer = False
                breaks = getJenksBreaks(
                    altered_generate_coding_position_list(
                        gene_of_interest,
                        limit_breast_cancer = limit_breast_cancer,
                        clinvar_df = clinvar_df
                    ),
                    int(n_breaks)
                )
                results_mega_dict[gene_of_interest][n_breaks]["breakpoints"] = breaks
                region_labels = []
                for index, row in gene_df.iterrows():
                    if row["Missense"] == 1:
                        coding_pos = row["coding_position"]
                        region = place_position(breaks, coding_pos)
                        region_labels.append(region + 1)
                    else:
                        region_labels.append(-1)

                gene_df["region_label"] = region_labels
                highest_regions = np.max(gene_df["region_label"])
                region_results = []
                for i in range(1, highest_regions + 1):
                    subset = gene_df.loc[
                        (gene_df["Deleterious"] == 0) &
                        ((gene_df["region_label"] == i) |
                        (gene_df["region_label"] == -1))
                    ]

                    n_carriers = len(subset.loc[subset["region_label"] == i])

                    cols_keeping = ["Missense", "breastcancer_age", "breastcancer"]
                    cols = copy.deepcopy(list(subset.columns))
                    for c in cols_keeping:
                        cols.remove(c)
                    subset = subset.drop(columns = cols)

                    fitter = CoxPHFitter()
                    fitter.fit(subset, event_col ="breastcancer", duration_col = "breastcancer_age")
                    regression_results = fitter.summary.loc["Missense"].to_dict()
                    regression_results["Region"] = i
                    regression_results["carriers"] = n_carriers
                    results_mega_dict[gene_of_interest][n_breaks][i] = regression_results
                    region_results.append(regression_results)


                region_results_df = pd.DataFrame(region_results)
                alt_df = copy.deepcopy(gene_df)
                alt_df = alt_df.loc[alt_df["Deleterious"] == 0]

                cols_keeping = ["breastcancer", "breastcancer_age"]
                for i in range(1, highest_regions + 1):
                    alt_df["Region " + str(i)] = [1 if x == i else 0 for x in list(alt_df["region_label"])]
                    cols_keeping.append("Region " + str(i))


                alt_df_filtered = alt_df[cols_keeping]
                fitter_2 = CoxPHFitter()
                fitter_2.fit(alt_df_filtered, event_col ="breastcancer", duration_col = "breastcancer_age")

                score_regions_included = fitter_2.log_likelihood_

                comp1 = copy.deepcopy(gene_df)
                comp1 = comp1.loc[comp1["Deleterious"] == 0]
                comp1 = comp1[["breastcancer", "breastcancer_age", "Missense"]]
                fitter_3 = CoxPHFitter()
                fitter_3.fit(comp1, event_col ="breastcancer", duration_col = "breastcancer_age")

                score_missense_included = fitter_3.log_likelihood_
                score_difference = score_regions_included - score_missense_included
                p = 1 - scipy.stats.chi2.cdf(2*score_difference, n_breaks-1)
                results_mega_dict[gene_of_interest][n_breaks]["missense_log_likelihood"] = score_missense_included
                results_mega_dict[gene_of_interest][n_breaks]["regions_log_likelihood"] = score_regions_included
                results_mega_dict[gene_of_interest][n_breaks]["chi_sq_p"] = p
                if p < min_p:
                    min_p = p
                    argmin_p = n_breaks
            except Exception as e:
                print("error...")
                print(n_breaks, gene_of_interest)
                print(e)

        results_of_het_test.append({"gene" :gene_of_interest, "min p" : min_p, "argmin breaks" :  argmin_p})
        filehandler = open("mega_results_checkppoint_092922.pickle","wb")
        pickle.dump(results_mega_dict, filehandler)




if __name__ == "__main__":
    global clinvar_df
    global patient_to_phenotype
    global results
    clinvar_df = load_clinvar_df()
    patient_to_phenotype = pd.read_csv(src_path + "data/updated_patient_to_phenotype_041721.csv", index_col = 0)
    results_dicts = get_regressions_results()
    results = pd.DataFrame(results_dicts)
    results = results.set_index("gene").sort_values(by = "Deleterious p")
    results.to_csv("regression_results_delet_removed_092922.csv")
    make_metrics_file()

