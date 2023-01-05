import pandas as pd
from utils import get_34_gene_list

def load_clinvar_df(fn = "/Users/jdf36/Desktop/bwh/cancer_regions/data/clinvar_2020-07.txt"):
    clinvar_df = pd.read_csv(fn, delimiter="\t")
    return clinvar_df

def is_breastcancer_phrase(phrase):
    if phrase == "" or phrase is None:
        return False

    phrase = phrase.lower().replace("\n", " ")
    if "breast" in phrase or "ovarian" in phrase:
        if "cancer" in phrase or "tumor" in phrase:
            return True
    return False


def genereate_coding_position_list(gene, limit_breast_cancer = True):
    to_return = []
    filtered_df = clinvar_df.loc[
        (clinvar_df["genesymbol"] == gene) &
        (clinvar_df["pathogenic"] == 1) &
        ## v removes "conflicting interpretatoins of pathogenicity "
        (~clinvar_df["all_significance"].str.contains("onflicting")) &
        ## v removes "uncertain * pathogenicity"
        (~clinvar_df["all_significance"].str.contains("ncertain"))
    ]
    if limit_breast_cancer:
        filtered_df["is_breastcancer"] = filtered_df["phenotypelist"].apply(
            lambda x : is_breastcancer_phrase(x)
        )
        filtered_df = filtered_df.loc[filtered_df["is_breastcancer"] == True]

    nucleotides = ["A", "T", "C", "G"]
    int_strings = [str(i) for i in range(0, 10)]
    for index, row in filtered_df.iterrows():
        if row["ref"] in nucleotides and row["alt"] in nucleotides:
            spl = row["name"].split(":c.")
            if len(spl) == 2:
                numeric_split = spl[1].split(">")[0]
                numeric_pos = ""
                i = 0
                while numeric_split[i] in int_strings:
                    numeric_pos = numeric_pos + numeric_split[i]
                    i+= 1
                if numeric_pos != "":
                    to_return.append(int(numeric_pos))

    return to_return

def altered_generate_coding_position_list(
    gene,
    limit_breast_cancer = True,
    clinvar_df = None,
    limit_pathogenic = True
):
    to_return = []
    if clinvar_df is None:
        clinvar_df = load_clinvar_df()
    filtered_df = clinvar_df.loc[(clinvar_df["genesymbol"] == gene)]
    if limit_breast_cancer:
        filtered_df["is_breastcancer"] = filtered_df["phenotypelist"].apply(
            lambda x : is_breastcancer_phrase(x)
        )
        filtered_df = filtered_df.loc[filtered_df["is_breastcancer"]]

    nucleotides = ["A", "T", "C", "G"]
    int_strings = [str(i) for i in range(0, 10)]
    for index, row in filtered_df.iterrows():
        vals_string = " ".join(list(map(lambda x : str(x), row.values)))
        if "splice" in vals_string:
            continue
        var_name = row["name"]
        if ("p." not in var_name) or ("Ter" in var_name):
            continue
        interpretations = row["all_significance"].split("|")
        if row["ref"] in nucleotides and row["alt"] in nucleotides:
            spl = row["name"].split(":c.")
            if len(spl) == 2:
                numeric_split = spl[1].split(">")[0]
                numeric_pos = ""
                i = 0
                while numeric_split[i] in int_strings:
                    numeric_pos = numeric_pos + numeric_split[i]
                    i+= 1
                if numeric_pos != "" and numeric_pos not in ["1", "2", "3"]:
                    for interpretation in interpretations:
                        interpretation = interpretation.lower()
                        if limit_pathogenic:
                            if ("pathogenic" in interpretation) and (not "conflicting" in interpretation) and (not "uncertain" in interpretation) and (not "no" in interpretation):
                                to_return.append(int(numeric_pos))
                        else:
                            to_return.append(int(numeric_pos))

    return to_return


def getJenksBreaks( dataList, numClass ):
    dataList.sort()
    mat1 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat1.append(temp)

    mat2 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat2.append(temp)

    for i in range(1,numClass+1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2,len(dataList)+1):
            mat2[j][i] = float('inf')

    v = 0.0
    for l in range(2,len(dataList)+1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1,l+1):
            i3 = l - m + 1
            val = float(dataList[i3-1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1

            if i4 != 0:
                for j in range(2,numClass+1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]

        mat1[l][1] = 1
        mat2[l][1] = v

    k = len(dataList)
    kclass = []
    for i in range(0,numClass+1):
        kclass.append(0)

    kclass[numClass] = float(dataList[len(dataList) - 1])

    countNum = numClass
    while countNum >= 2:
        id = int((mat1[k][countNum]) - 2)

        kclass[countNum - 1] = dataList[id]
        k = int((mat1[k][countNum] - 1))
        countNum -= 1

    return kclass

def getGVF( dataList, numClass ):
        """
        The Goodness of Variance Fit (GVF) is found by taking the
        difference between the squared deviations
        from the array mean (SDAM) and the squared deviations from the
        class means (SDCM), and dividing by the SDAM
        """
        breaks = getJenksBreaks(dataList, numClass)
        dataList.sort()
        listMean = sum(dataList)/len(dataList)
        SDAM = 0.0
        for i in range(0,len(dataList)):
                sqDev = (dataList[i] - listMean)**2
                SDAM += sqDev

        SDCM = 0.0
        for i in range(0,numClass):
                if breaks[i] == 0:
                        classStart = 0
                else:
                        classStart = dataList.index(breaks[i])
                        classStart += 1
                classEnd = dataList.index(breaks[i+1])

                classList = dataList[classStart:classEnd+1]

                classMean = sum(classList)/len(classList)
                preSDCM = 0.0
                for j in range(0,len(classList)):
                         sqDev2 = (classList[j] - classMean)**2
                         preSDCM += sqDev2
                SDCM += preSDCM
        return (SDAM - SDCM)/SDAM

if __name__ == "__main__":
    global clinvar_df
    clinvar_df = load_clinvar_df()
    genes = [
        "ATM",
        "BARD1",
        "BRCA1",
        "BRCA2",
        "CHEK2",
        "MSH6",
        "PALB2",
        "RAD51C",
        "RAD51D",
        "TP53"
    ]
    non_bc_genes = ["MSH6"]
    for g in genes:
        limit_breast_cancer = True
        if g in non_bc_genes:
            limit_breast_cancer = False
        breaks = getJenksBreaks(altered_generate_coding_position_list(
                g,
                limit_breast_cancer = limit_breast_cancer,
                clinvar_df = clinvar_df
            ),
            15
        )
