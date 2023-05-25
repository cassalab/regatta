import pandas as pd
import pickle 
import glob
import gzip
import numpy as np
import scipy.stats
from pyliftover import LiftOver

gene_to_end_position = {
    "ABRAXAS1": 1227,
    "ATM" : 9168,
    "AKT1" : 1440,
    "BABAM2" : 1149,
    "BRIP1" : 3747,
    "BARD1" : 2331,
    "BRCA1" : 5589,
    "BRCA2" : 10254,
    "CDH1" : 2646,
    "CHEK2" : 1629,
    "EPCAM" : 942,
    "FANCC" : 1674,
    "FANCM": 6144,
    "GEN1" : 2724,
    "MEN1" : 1845,
    "MLH1" : 2268,
    "MRE11" : 2124,
    "MSH2" : 2802,
    "MSH6" : 4080,
    "MUTYH" : 1638,
    "NBN" : 2262,
    "NF1" : 8517,
    "PALB2" : 3558,
    "PIK3CA" : 3204,
    "PMS2" : 2586,
    "PTEN" : 1209,
    "RAD50" : 1312,
    "RAD51C" : 1128,
    "RAD51D" : 984,
    "RECQL" : 1947,
    "RINT1" : 2376,
    "STK11" : 1299,
    "TP53" : 1179,
    "XRCC2" : 840
}

TP53_hr = [[338, 388], [659,725], [775, 826], [887, 1182]]
TP53_lr = [[105,337],[389,541],[596, 658],[726,774],[827,886]]

PALB2_hr = [[0, 107], [2559, 2673]]
PALB2_lr = [[2674, 3558]]

BRCA1_hr = [[0,440], [5365, 5589]]
BRCA1_lr = [[2596, 3178], [4327, 4483], [4992, 5257]]

BRCA2_hr = [[1799,2622], [3504, 4589], [7879, 8083], [8243, 8524], [8851, 10254]]
BRCA2_lr = [[229, 830], [4590, 5960]]

MSH6_hr = [[1445, 1753], [2117, 2341], [3656, 4080]]
MSH6_lr = [[431, 1345], [1754, 2116], [2927, 3226]]

CHEK2_hr = [[470, 556], [715,1182]]
CHEK2_lr = [[254, 469], [1183,1282]]

ATM_hr = [[5228,7996]]
ATM_lr = [[1, 1546], [3284,3872], [7997,8292], [8737, 9168]]

gene_to_high = {
    "BRCA1" : BRCA1_hr,
    "BRCA2" : BRCA2_hr,
    "TP53" : TP53_hr,
    "PALB2" : PALB2_hr,
    "ATM" : ATM_hr,
    "CHEK2" : CHEK2_hr,
    "MSH6" : MSH6_hr
}

gene_to_low = {
    "BRCA1" : BRCA1_lr,
    "BRCA2" : BRCA2_lr,
    "TP53" : TP53_lr,
    "PALB2" : PALB2_lr,
    "ATM" : ATM_lr,
    "CHEK2" : CHEK2_lr,
    "MSH6" : MSH6_lr
}

def unpickle_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data

def in_region_list(pos, reg_list):
    for r in reg_list:
        if pos >= r[0] and pos <= r[1]:
            return True 
    return False 

lengths = []
total_over_90 = []
total_over_95 = []
total_obs = []

gene_totals_vals = []
gene_hr_lr_vals_90 = []
gene_hr_lr_vals_95 = []


def main():
   
   lo = LiftOver('hg19', 'hg38')
   
   for gene in ["ATM", "BRCA1", "BRCA2", "TP53", "MSH6", "CHEK2", "PALB2"]:
        # print("$$$$$$$$$$$$$$$$$$$$$", gene)
        files = glob.glob(gene  + "*exon*.pickle")

        df_list = []

        for f in files:
            df = pd.DataFrame(unpickle_file(f))
            df_list.append(df)

        df = pd.concat(df_list, axis=0, ignore_index=True)
        merged_df = df.loc[~df["cds_start"].isna()].sort_values(by = "cds_start")
        
        observed_nts = set(merged_df["cds_start"])
        for i in np.arange(1, np.max(merged_df["cds_start"]) + 1):
            assert i in observed_nts

        # print(np.max(list(observed_nts)), gene_to_end_position[gene])
        
        target_chrom = merged_df["chrom"].values[0]
        max_g_pos = np.max(merged_df["genomic_pos"].values)
        min_g_pos =  np.min(merged_df["genomic_pos"].values)
        # print(max_g_pos, min_g_pos)
        counter = 0
    #     print(gene, target_chrom, max_g_pos, min_g_pos)
        in_range_genes = set()
        ccrs_dict = {}
        with gzip.open('ccrs.autosomes.v2.20180420.bed.gz','rt') as fin:        
            for line in fin: 
                if counter == 0:
    #                 print(line.split("\t")) #header line 
                    counter += 1
                    continue 
                vals = line.split("\t")
                chrom = int(vals[0])
                if chrom == target_chrom:
                    start = int(vals[1])
                    end = int(vals[2])
                    try:
                        start = lo.convert_coordinate(f'chr{chrom}', start)[0][1]
                        end = lo.convert_coordinate(f'chr{chrom}', end)[0][1]
                    except:
                        # print("error with " , vals[ : 3] )
                        continue 

                    ccr_pct = float(vals[3])
                    if start >= min_g_pos and end <= max_g_pos:
                        for p in range(start, end):
                            ccrs_dict[p] = ccr_pct
                            in_range_genes.add(vals[4])
                counter += 1
        fin.close()
        
        # print("ccrs_dict", len(ccrs_dict))
        observed_nts = set(merged_df["cds_start"])
        for i in np.arange(1, np.max(merged_df["cds_start"]) + 1):
            assert i in observed_nts

        # print(np.max(list(observed_nts)), gene_to_end_position[gene])
        g_pos_to_c_pos = dict(zip(merged_df["genomic_pos"].values, merged_df["cds_start"].values))
        c_pos_to_ccr = {}
        for g, c in g_pos_to_c_pos.items():
            if g in ccrs_dict.keys():
                c_pos_to_ccr[c] = ccrs_dict[g]


        total_gene_over_95 = list(filter(lambda x : x >= 95, list(c_pos_to_ccr.values())))
        total_gene_over_90 = list(filter(lambda x : x >= 90, list(c_pos_to_ccr.values())))

        gene_totals_vals.append([gene, (len(total_gene_over_90) / len(c_pos_to_ccr)),  (len(total_gene_over_95) / len(c_pos_to_ccr))])

        total_over_90.append(len(total_gene_over_90))
        total_over_95.append(len(total_gene_over_95))
        total_obs.append(len(c_pos_to_ccr))

        print("running....", ">= 90", np.sum(total_over_90) / np.sum(total_obs), ">= 95", np.sum(total_over_95) / np.sum(total_obs))

      
        high_lst = []
        low_lst = []
        for c_pos, ccr in c_pos_to_ccr.items():
            if in_region_list(c_pos , gene_to_high[gene]):
                high_lst.append(ccr)
            if in_region_list(c_pos , gene_to_low[gene]):
                low_lst.append(ccr)

        # print("----------- Above 90 -----------")
        high_lst_2 = list(filter(lambda x : x >= 90, high_lst))
        low_lst_2 = list(filter(lambda x : x >= 90, low_lst))

        gene_hr_lr_vals_90.append([gene, 90, round(len(high_lst_2)/ len(high_lst), 3), round(len(low_lst_2)/ len(low_lst), 3)])

        
        high_lst_2 = list(filter(lambda x : x >= 95, high_lst))
        low_lst_2 = list(filter(lambda x : x >= 95, low_lst))
        gene_hr_lr_vals_95.append([gene, 95, round(len(high_lst_2)/ len(high_lst), 3), round(len(low_lst_2)/ len(low_lst), 3)])
        lengths.append(len(c_pos_to_ccr))


if __name__ == "__main__":
    main()
    df1 = pd.DataFrame(gene_totals_vals, columns = ["gene", "fraction bp >= 90 CCR", "fraction bp >= 95 CCR"])
    df1.to_csv("gene_total_constrint_values.csv", index = False)
    total_constraint = gene_hr_lr_vals_90 + gene_hr_lr_vals_95
    df2 = pd.DataFrame(total_constraint, columns = ["gene", "constraint value", "fraction HRR bp >= constraint value",  "fraction LRR bp >= constraint value"])
    df2.to_csv("gene_total_constrint_values_HRR_LRR.csv", index = False)
