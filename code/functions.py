import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import json
from typing import List, Dict


def finding_unique_L_SC1():  # gov. expenditures
    unique_values = []
    elements = os.listdir("../Open-LPP-data/base_complete")
    for i in range(int(len(elements) / 2)):
        df = pd.read_csv(
            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i+2014)}.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )
        values = df["L_SC1"].unique().tolist()
        for value in values:
            if value in unique_values:
                pass
            else:
                unique_values.append(value)

    return unique_values


def finding_unique_L_CODE_LPP():  # complementary insurances expenditures
    unique_values = []
    elements = os.listdir("../Open-LPP-data/base_complementaire")
    for i in range(int(len(elements) / 2)):
        df = pd.read_csv(
            f"../Open-LPP-data/base_complementaire/NB_{(i+2014)}_lpp.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )
        values = df["L_CODE_LPP"].unique().tolist()
        for value in values:
            if value in unique_values:
                pass
            else:
                unique_values.append(value)

    return unique_values


def get_L_SC1_SC2_LPP_gov_exp():
    L_SC1 = {}
    unique_values = []
    elements = os.listdir("../Open-LPP-data/base_complete")

    # On parcourt chaque fichier (en supposant qu'il y en a plusieurs)
    for i in range(int(len(elements) / 2)):
        df = pd.read_csv(
            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i + 2014)}.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )
        values = df["L_SC1"].unique().tolist()

        # Récupérer les valeurs uniques de L_SC1
        for value in values:  # L_SC1 step
            if value not in unique_values:
                unique_values.append(value)

        for value in unique_values:  # L_SC2 step
            if value not in L_SC1:
                L_SC1[value] = {}

            df_filtered = df[df["L_SC1"] == str(value)]

            for j in range(len(df_filtered)):
                # Récupérer le titre L_SC2 correctement
                L_SC2_title = str(
                    df_filtered.iloc[j]["L_SC2"]
                ).strip()  # Assure-toi que c'est une chaîne
                # print(f"L_SC2_title : {L_SC2_title}")
                if L_SC2_title not in L_SC1[value]:
                    L_SC1[value][L_SC2_title] = []  # Initialisation avec une liste

                # Ajoutons les codes LPP
                code_LPP_title = str(
                    df_filtered.iloc[j]["L_CODE_LPP"]
                ).strip()  # Assure-toi que c'est une chaîne
                # print(f"code_LPP_title : {code_LPP_title}")
                if code_LPP_title not in L_SC1[value][L_SC2_title]:
                    L_SC1[value][L_SC2_title].append(code_LPP_title)
        print(f"{i+2014} fini")
    return L_SC1


def get_potential_LPP_only():  # to apply on the result of get_L_SC1_SC2_LPP_gov_exp(), you can get all potentials data that is under 100% Santé plan
    deleted_L_SC1 = [
        "ACCESSOIRES DE PRODUITS INSCRITS AU TITRE III",
        "ADJONCTIONS, OPTIONS ET REPARATIONS APPLICABLES AUX FAUTEUILS ROULANTS",
        "APPAREIL GENERATEUR D AEROSOL",
        "ARTICLES POUR PANSEMENTS, MATERIELS DE CONTENTION",
        "CODES ARRIVES A ECHEANCE",
        "DISPOISTIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO_VASCULAIRE",
        "DISPOSITIFS MEDICAUX IMPLANTABLES ACTIFS",
        "DISPOSITIFS MEDICAUX UTILISES DANS LE SYST CARDIO-VASCULAIRE",
        "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME GASTRO-INTESTINAL",
        "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME URO-GENITAL",
        "DISPOSITIFS MEDICAUX UTILISES EN NEUROLOGIE",
        "DISPOSITIFS MEDICAUX UTILISES EN ONCOLOGIE",
        "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPE",
        "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPES",
        "DM, MATERIELS ET PRODUITS POUR LE TRAITEMENT DE PATHOLOGIES SPECIFIQUES",
        "DMI ISSUS DE DERIVES ORIGINE ANIMALE NON VIABLES OU EN COMPORTANT",
        "FAUTEUILS ROULANTS",
        "IMPLANTS ISSUS DE DERIVES HUMAINS-GREFFONS",
        "IMPLANTS ISSUS DE DERIVES HUMAINS_GREFFONS",
        "ORTHESES",
        "ORTHESES (PETIT APPAREILLAGE) (CHAP.1)",
        "ORTHOPROTHESES(CHAP.7)",
        "PODO-ORTHESES",
        "PODO_ORTHESES",
        "PROTHESES EXTERNES NON ORTHOPEDIQUES",
        "VEHICULES DIVERS",
    ]

    with open("../data/L_SC1_SC2_LPP_gov_exp_raw.json", "r") as file:
        data = json.load(file)
    for title in deleted_L_SC1:
        del data[title]

    supp_list = []
    for title in data["DMI D ORIGINE SYNTHETIQUE"]:
        if title != "PROTHESES AUDITIVES OSTEO-INTEGREES":
            supp_list.append(title)
        else:
            pass

    for title in supp_list:
        del data["DMI D ORIGINE SYNTHETIQUE"][title]

    return data


def adjusted_price(
    HICP_df, initial_price, year_initial_price
):  # useable with HICP rates, in the file "data/HICP"
    HICP_df = HICP_df[HICP_df["TIME PERIOD"] >= year_initial_price]
    for i in range(len(HICP_df)):
        initial_price = initial_price * (1 + (HICP_df.iloc[i, 2] / 1000))
    return initial_price


# example :
# optical_HICP = pd.read_csv("../data/HICP/HICP-Corrective-eye-glasses-and-contact-lenses-France-Annual-parts-per-1000.csv")
# adjusted = adjusted_price(optical_HICP, sum, 2023)


def gov_exp(
    inflation_adjustment: bool, sector: str, mask: Dict[str, List[str]], indent: int
):  # but this is exclusive mask, I have to do sth that can put a OR condition on the masks
    expenditures = (
        {}
    )  # we can take a dict as an arg, and iterate on it in a loop to filter 1 time the data with 1 mask, a 2nd time with a 2nd mask, ... until we reach the len of the dict. {"L_SC1":["PROTHESES", (contains or ==)], "L_SC1":["VERRE",(contains or ==)], "L_SC2":["LUNETTES",(contains or ==)]}
    if sector not in ["optical", "dental", "hearing", "all"]:
        raise ValueError(
            "'sector' argument has to be one of the four following : 'optical', 'dental', 'hearing', 'all'"
        )
    if mask == {}:
        pass
    else:
        for k in mask:
            if mask[k][0] not in ["equality", "contains"]:
                raise ValueError(
                    "The first value in mask's values has to be whether 'equality' or 'contains'"
                )
            if mask[k][1] not in ["L_SC1", "L_SC2", "L_CODE_LPP"]:
                raise ValueError(
                    "The second value in mask's values has to be 'L_SC1', 'L_SC2' or 'L_CODE_LPP'"
                )
            if len(mask) == 1:
                pass
            elif mask[k][2] not in ["and", "or"]:
                raise ValueError(
                    "The third value in mask's values has to be whether 'and' or 'or'"
                )

    elements = os.listdir("../Open-LPP-data/base_complete")
    nb_LPP_codes = {}
    nb_refunds = {}
    refund_rate = {}
    base = {}
    for i in range(int(len(elements) / 2) - indent):
        df = pd.read_csv(
            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i+2014+indent)}.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )
        final_mask = None
        if mask == {}:
            pass
        else:
            for filter in mask:
                if (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC1"):
                    new_mask = df[mask[filter][1]] == filter
                elif (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC2"):
                    new_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "equality") & (
                    mask[filter][1] == "L_CODE_LPP"
                ):
                    new_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC1"):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC2"):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                elif (mask[filter][0] == "contains") & (
                    mask[filter][1] == "L_CODE_LPP"
                ):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                # print(i + 2014+indent)
                if final_mask is None:
                    final_mask = new_mask
                else:
                    if mask[filter][2] == "and":
                        final_mask = final_mask & new_mask
                    else:
                        final_mask = final_mask | new_mask

            df = df[final_mask]

        df = pd.DataFrame(
            {
                "L_SC1": df["L_SC1"],
                "L_SC2": df["L_SC2"],
                "CODE_LPP": df["CODE_LPP"],
                "L_CODE_LPP": df["L_CODE_LPP"],
                "Quantity": df["QTE"],
                "Financing": df["REM"],
                "BASE": df["BSE"],
            }
        )

        #df = df[df["L_CODE_LPP"].str.contains("PILES", case=False, na=False)]

        #print((i+2014+indent))
        #dfprinted = pd.concat([df["L_CODE_LPP"], df["Quantity"]], axis=1)
        #return dfprinted

        #print(df["L_CODE_LPP"].unique())  # pour vérifier manuellement si les filtres sont bien appliqués

        df.reset_index(inplace=True)
        df.drop(columns="index", inplace=True)
        sum = df["Financing"].sum()
        rate = []
        key = str(i + 2014 + indent)

        base_sum = df["BASE"].sum()
        base[key] = base_sum

        if len(df) == 0:
            refund_rate[key] = 0
        else:
            for i in range(len(df)):
                if (
                    pd.isna(df.loc[i, "Financing"])
                    or pd.isna(df.loc[i, "BASE"])
                    or df.loc[i, "BASE"] == 0
                ):
                    continue
                else:
                    rate.append(df.loc[i, "Financing"] / df.loc[i, "BASE"])
            refund_rate[key] = np.mean(rate)
        # print(f"key : {key}")

        nb_LPP_codes[key] = len(df["CODE_LPP"].unique().tolist())
        nb_refunds[key] = df["Quantity"].sum()

        if inflation_adjustment == True:
            if sector == "optical":
                optical_HICP = pd.read_csv(
                    "../data/HICP/HICP-Corrective-eye-glasses-and-contact-lenses-France-Annual-parts-per-1000.csv"
                )
                expenditures[key] = adjusted_price(optical_HICP, sum, i + 2014 + indent)
            elif sector == "hearing":
                print(
                    "Hearing HICP is neagligeable face to the amount of money and the trend is the same whether we take HICP in count or not."
                )
            elif sector == "all":
                pass
        else:
            expenditures[key] = sum

    return [expenditures, nb_LPP_codes, nb_refunds, refund_rate, base]

# example : whole_hearing = gov_exp(inflation_adjustment=False, sector="hearing", mask={"AUDIOPROTHESES":["contains", "L_SC1", "or"]}, indent=4)

def normalized_by_mean(list: List[int]):
    normalized_list = []
    mean = np.mean(list)
    for int in list:
        int = int / mean
        normalized_list.append(int)

    return normalized_list

# problem with JSON serializing (due to int64 format)
def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def gov_exp_by_age(
    inflation_adjustment: bool, sector: str, mask: Dict[str, List[str]], indent: int
):  # but this is exclusive mask, I have to do sth that can put a OR condition on the masks
    expenditures = (
        {}
    )  # we can take a dict as an arg, and iterate on it in a loop to filter 1 time the data with 1 mask, a 2nd time with a 2nd mask, ... until we reach the len of the dict. {"L_SC1":["PROTHESES", (contains or ==)], "L_SC1":["VERRE",(contains or ==)], "L_SC2":["LUNETTES",(contains or ==)]}
    if sector not in ["optical", "dental", "hearing", "all"]:
        raise ValueError(
            "'sector' argument has to be one of the four following : 'optical', 'dental', 'hearing', 'all'"
        )
    if mask == {}:
        pass
    else:
        for k in mask:
            if mask[k][0] not in ["equality", "contains"]:
                raise ValueError(
                    "The first value in mask's values has to be whether 'equality' or 'contains'"
                )
            if mask[k][1] not in ["L_SC1", "L_SC2", "L_CODE_LPP"]:
                raise ValueError(
                    "The second value in mask's values has to be 'L_SC1', 'L_SC2' or 'L_CODE_LPP'"
                )
            if len(mask) == 1:
                pass
            elif mask[k][2] not in ["and", "or"]:
                raise ValueError(
                    "The third value in mask's values has to be whether 'and' or 'or'"
                )

    expend_by_age = {}

    elements = os.listdir("../Open-LPP-data/base_complete")
    for i in range(int(len(elements) / 2) - indent):
        age = {}
        df = pd.read_csv(
            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i+2014+indent)}.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )

        final_mask = None
        if mask == {}:
            pass
        else:
            for filter in mask:
                if (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC1"):
                    new_mask = df[mask[filter][1]] == filter
                elif (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC2"):
                    new_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "equality") & (
                    mask[filter][1] == "L_CODE_LPP"
                ):
                    new_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC1"):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC2"):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                elif (mask[filter][0] == "contains") & (
                    mask[filter][1] == "L_CODE_LPP"
                ):
                    regex_pattern = rf"\b{filter}\b"
                    new_mask = df[mask[filter][1]].str.contains(
                        regex_pattern, case=False, na=False
                    )

                # print(i + 2014+indent)
                if final_mask is None:
                    final_mask = new_mask
                else:
                    if mask[filter][2] == "and":
                        final_mask = final_mask & new_mask
                    else:
                        final_mask = final_mask | new_mask

            df = df[final_mask]

        df = pd.DataFrame(
            {
#                "L_SC1": df["L_SC1"],
#                "L_SC2": df["L_SC2"],
#                "CODE_LPP": df["CODE_LPP"],
#                "L_CODE_LPP": df["L_CODE_LPP"],
                "Quantity": df["QTE"],
                "Financing": df["REM"],
#                "BASE": df["BSE"],
                "AGE":df["AGE"]
            }
        )

        sum0_20 = df[df["AGE"]==0]["Financing"].sum()
        len0_20 = len(df[df["AGE"]==0])
        qte0_20 = df[df["AGE"]==0]["Quantity"].sum()

        sum20_40 = df[df["AGE"]==20]["Financing"].sum()
        len20_40 = len(df[df["AGE"]==20])
        qte20_40 = df[df["AGE"]==20]["Quantity"].sum()

        sum40_60 = df[df["AGE"]==40]["Financing"].sum()
        len40_60 = len(df[df["AGE"]==40])
        qte40_60 = df[df["AGE"]==40]["Quantity"].sum()

        sum60_100 = df[df["AGE"]==60]["Financing"].sum()
        len60_100 = len(df[df["AGE"]==60])
        qte60_100 = df[df["AGE"]==60]["Quantity"].sum()

        #sum_unknown = df[df["AGE"]==99]["Financing"].sum()
        #lenunknown = len(df[df["AGE"]==99])
        #qte_unknown = df[df["AGE"]==99]["Quantity"].sum()

        key = str(i + 2014)

        sum_list = [sum0_20, sum20_40, sum40_60, sum60_100]
        len_list = [len0_20, len20_40, len40_60, len60_100]
        qte_list = [qte0_20, qte20_40, qte40_60, qte60_100]

        for i in range(4):
            #if i == 4:
            #    age["unknown"] = [lenunknown, sum_unknown], really non-significant
            #else:
            age[f"{i*20}-{(i+1)*20}"] = [len_list[i], sum_list[i], qte_list[i]]

        expend_by_age[key] = age

    #expand_by_age={"year":{"0-20":[len, expenditures]}}

    return expend_by_age


def getting_df_with_age(with_interaction = True):
    with open("../data/results/L_SC1_SC2_LPP_gov_exp_raw.json") as f:
        L_SC1 = json.load(f)

    list_group=list(L_SC1.keys())

    to_remove = ['CODES ARRIVES A ECHEANCE', 
    'ACCESSOIRES DE PRODUITS INSCRITS AU TITRE III', 
    "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME GASTRO-INTESTINAL", 
    "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME URO-GENITAL", 
    "DISPOSITIFS MEDICAUX UTILISES EN NEUROLOGIE", 
    "DISPOSITIFS MEDICAUX UTILISES EN ONCOLOGIE", 
    "DM, MATERIELS ET PRODUITS POUR LE TRAITEMENT DE PATHOLOGIES SPECIFIQUES", 
    'DISPOSITIFS MEDICAUX UTILISES DANS LE SYST CARDIO-VASCULAIRE',
    "DISPOISTIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO_VASCULAIRE",
    'DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPES',
    "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPE",
    'DMI ISSUS DE DERIVES ORIGINE ANIMALE NON VIABLES OU EN COMPORTANT', 
    'IMPLANTS ISSUS DE DERIVES HUMAINS_GREFFONS',
    "IMPLANTS ISSUS DE DERIVES HUMAINS-GREFFONS",
    'PODO_ORTHESES',
    "PODO-ORTHESES",
    ]
    #problem was that some elements was missing with these title, or the expends were constant.

    for element in to_remove:
        if element in list_group:
            list_group.remove(element)

    #count = 0
    variable_dict = {}
    dfs = [] # Liste pour stocker les DataFrames, plus rapide que de concat à chaque itération
    #years = [i+2014 for i in range(10)]

    for title in list_group:
        print(title)
        variable_name = title
        #print(variable_name, type(variable_name))
        if title in ["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEURS"]:
            continue

        elif title == "AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR":
            variable_dict[variable_name] = gov_exp_by_age(
                inflation_adjustment=False,
                sector="all",
                mask={title: ["equality", "L_SC1", "or"], "AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEURS":["equality", "L_SC1", "or"]},
                indent=0,
            )

        #elif title == "DISPOISTIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO_VASCULAIRE":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO-VASCULAIRE":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPE":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPES":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "IMPLANTS ISSUS DE DERIVES HUMAINS-GREFFONS":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "IMPLANTS ISSUS DE DERIVES HUMAINS_GREFFONS":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "PODO-ORTHESES":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "PODO_ORTHESES":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        else:
            variable_dict[variable_name] = gov_exp_by_age(
                inflation_adjustment=False,
                sector="all",
                mask={title: ["equality", "L_SC1", "and"]},
                indent=0,
            )

        print(f"got all '{title} expenditures'")

        data = []

        for year, age_data in variable_dict[variable_name].items():
            for age_range, values in age_data.items():
                data.append({
                    "year": year,
                    "age_range": age_range,
                    #"treatment": treatment[k],
                    "expenditures": values[1],
                    "quantities": values[2],
                    "group":title,
                })

        df = pd.DataFrame(data)

        print(df)

        print(f"got {title} df")
        dfs.append(df)

    df_final = pd.concat(dfs, axis=0)
        #if i == 0:
        #    df_final = df
        #    i+=1
        #else:
        #    df_final = pd.concat([df_final, df], axis=0)

    df_final = pd.get_dummies(df_final, columns=['group'], prefix='', prefix_sep='')
    df_final = pd.get_dummies(df_final, columns=['year'], prefix='', prefix_sep='')
    df_final = pd.get_dummies(df_final, columns=['age_range'], prefix='', prefix_sep='')

    if with_interaction == True:
        for col in df_final.filter(like="20").columns:
            df_final[f"interact. audio x year{col}"] = df_final["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR"] * df_final[col]
            df_final[f"interact. optique x year{col}"] = df_final["OPTIQUE MEDICALE"] * df_final[col]
            df_final[f"interact. (audio+optical) x year{col}"] = (df_final["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR"]+df_final["OPTIQUE MEDICALE"]) * df_final[col]
        return df_final

    else:
        return df_final
    


def getting_df_without_age(with_interaction = True):
    with open("../data/results/L_SC1_SC2_LPP_gov_exp_raw.json") as f:
        L_SC1 = json.load(f)

    list_group=list(L_SC1.keys())

    to_remove = ['CODES ARRIVES A ECHEANCE', 
    'ACCESSOIRES DE PRODUITS INSCRITS AU TITRE III', 
    "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME GASTRO-INTESTINAL", 
    "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME URO-GENITAL", 
    "DISPOSITIFS MEDICAUX UTILISES EN NEUROLOGIE", 
    "DISPOSITIFS MEDICAUX UTILISES EN ONCOLOGIE", 
    "DM, MATERIELS ET PRODUITS POUR LE TRAITEMENT DE PATHOLOGIES SPECIFIQUES", 
    'DISPOSITIFS MEDICAUX UTILISES DANS LE SYST CARDIO-VASCULAIRE',
    "DISPOISTIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO_VASCULAIRE",
    'DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPES',
    "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPE",
    'DMI ISSUS DE DERIVES ORIGINE ANIMALE NON VIABLES OU EN COMPORTANT', 
    'IMPLANTS ISSUS DE DERIVES HUMAINS_GREFFONS',
    "IMPLANTS ISSUS DE DERIVES HUMAINS-GREFFONS",
    'PODO_ORTHESES',
    "PODO-ORTHESES",
    ]
    #problem was that some elements was missing with these title, or the expends were constant.

    for element in to_remove:
        if element in list_group:
            list_group.remove(element)

    #count = 0
    variable_dict = {}
    dfs = [] # Liste pour stocker les DataFrames, plus rapide que de concat à chaque itération
    #years = [i+2014 for i in range(10)]

    for title in list_group:
        print(title)
        variable_name = title
        #print(variable_name, type(variable_name))
        if title in ["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEURS"]:
            continue

        elif title == "AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR":
            variable_dict[variable_name] = gov_exp(
                inflation_adjustment=False,
                sector="all",
                mask={title: ["equality", "L_SC1", "or"], "AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEURS":["equality", "L_SC1", "or"]},
                indent=0,
            )

        #elif title == "DISPOISTIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO_VASCULAIRE":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "DISPOSITIFS MEDICAUX UTILISES DANS LE SYSTEME CARDIO-VASCULAIRE":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPE":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "DM DE MAINTIEN A DOMICILE ET D AIDE A LA VIE POUR MALADES ET HANDICAPES":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "IMPLANTS ISSUS DE DERIVES HUMAINS-GREFFONS":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "IMPLANTS ISSUS DE DERIVES HUMAINS_GREFFONS":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        #elif title == "PODO-ORTHESES":
        #    variable_dict[variable_name] = gov_exp_by_age(
        #        inflation_adjustment=False,
        #        sector="all",
        #        mask={title: ["equality", "L_SC1", "or"], "PODO_ORTHESES":["equality", "L_SC1", "or"]},
        #        indent=0,
        #    )

        else:
            variable_dict[variable_name] = gov_exp(
                inflation_adjustment=False,
                sector="all",
                mask={title: ["equality", "L_SC1", "and"]},
                indent=0,
            )

        print(f"got all '{title} expenditures'")

        #if group_name == "optical":
        #    treatment = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        #elif group_name == "AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR":
        #    treatment = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        #else:
        #    treatment = [0 for j in range(10)]

        data = {
            "year": variable_dict[variable_name][0].keys(),
            "expenditures": variable_dict[variable_name][0].values(),
            "nb_code_LPP": variable_dict[variable_name][1].values(),
            "quantities": variable_dict[variable_name][2].values(),
            "group":title,
        }

        df = pd.DataFrame(data)

        print(df)

        print(f"got {title} df")
        dfs.append(df)

    df_final = pd.concat(dfs, axis=0)
        #if i == 0:
        #    df_final = df
        #    i+=1
        #else:
        #    df_final = pd.concat([df_final, df], axis=0)

    df_final = pd.get_dummies(df_final, columns=['group'], prefix='', prefix_sep='')
    df_final = pd.get_dummies(df_final, columns=['year'], prefix='', prefix_sep='')

    if with_interaction == True:
        for col in df_final.filter(like="20").columns:
            df_final[f"interact. audio x year{col}"] = df_final["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR"] * df_final[col]
            df_final[f"interact. optique x year{col}"] = df_final["OPTIQUE MEDICALE"] * df_final[col]
            df_final[f"interact. (audio+optical) x year{col}"] = (df_final["AUDIOPROTHESES ET ENTRETIEN, REPARATIONS ET ACCESSOIRES POUR PROCESSEUR"]+df_final["OPTIQUE MEDICALE"]) * df_final[col]
        return df_final

    else:
        return df_final
    

def gov_exp_dental(cent_santé=False, to_keep=[]):
    elements = os.listdir("../Open-CCAM-data")
    unique_values = []

    expenditures = {}
    #nb_LPP_codes = {}
    nb_refunds = {}
    refund_rate = {}
    base = {}

    for i in range(len(elements)):
        if (i+2015) in [2015, 2016, 2017, 2018, 2019]:
            df=pd.read_excel(f'../Open-CCAM-data/{i+2015}_CCAM.xls', sheet_name=1)
        else:
            df=pd.read_excel(f'../Open-CCAM-data/{i+2015}_CCAM.xlsx', sheet_name=1)

        #print(i+2015)
        
        if cent_santé == True:
            mask = df["Libellé long"].isin(to_keep)
            df = df[mask]

        df = pd.DataFrame(
            {
                "Complet": df.iloc[:, 2],
                "Lib. long" : df.iloc[:,1],
                #"Cat. acte": df.iloc[:, -2],
                #"Sous-cat. acte": df.iloc[:, -1],
                "QTE": df.iloc[:, -5],
                "REM": df.iloc[:, -3],
                "BSE": df.iloc[:, -4],
            }
        )
        
        df.reset_index(inplace=True)
        df.drop(columns="index", inplace=True)
        df = df.dropna()
        df = df.reset_index(drop=True)

        key = str(i+2015)
        
        sum = df["REM"].sum()
        base_sum = df["BSE"].sum()

        expenditures[key] = sum
        base[key] = base_sum

        rate = []
        if len(df) == 0:
            refund_rate[key] = 0
        else:
            for i in range(len(df)):
                if (
                    pd.isna(df.loc[i, "REM"])
                    or pd.isna(df.loc[i, "BSE"])
                    or df.loc[i, "BSE"] == 0
                ):
                    continue
                else:
                    rate.append(df.loc[i, "REM"] / df.loc[i, "BSE"])
            refund_rate[key] = np.mean(rate)
        # print(f"key : {key}")

        #nb_LPP_codes[key] = len(df["CODE_LPP"].unique().tolist())
        nb_refunds[key] = df["QTE"].sum()

    return [expenditures, nb_refunds, refund_rate, base]