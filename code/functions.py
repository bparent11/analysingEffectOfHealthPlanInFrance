import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import json
from typing import List, Dict


def finding_unique_L_SC1():
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


# def gov_optical_exp(inflation_adjustment): #maybe we can take final_mask as an argument in the function, but I got problem last time I tried it.
#    optical_expenditures = {} #we can take a dict as an arg, and iterate on it in a loop to filter 1 time the data with 1 mask, a 2nd time with a 2nd mask, ... until we reach the len of the dict. {"L_SC1":["PROTHESES", (contains or ==)], "L_SC1":["VERRE",(contains or ==)], "L_SC2":["LUNETTES",(contains or ==)]}
#    elements = os.listdir("../Open-LPP-data/base_complete")
#    for i in range(int(len(elements) / 2)):
#        df = pd.read_csv(
#            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i+2014)}.CSV",
#            encoding="ISO-8859-1",
#            sep=";",
#        )
#        final_mask = (df["L_SC1"] == "PROTHESES OCULAIRES ET FACIALES") & (
#            df["L_SC2"].str.contains("OCULAIRES|FRAIS")
#        )
#        # print(i+2014)
#        df = df[final_mask]
#        df = pd.DataFrame(
#            {
#                "L_SC2": df["L_SC2"],
#                "CODE_LPP": df["CODE_LPP"],
#                "Quantity": df["QTE"],
#                "Financing": df["REM"],
#            }
#        )
#        df.reset_index(inplace=True)
#        df.drop(columns="index", inplace=True)
#        df["Total"] = df["Quantity"] * df["Financing"]
#        sum = df["Total"].sum()
#        key = str(i + 2014)
#
#        if inflation_adjustment == True:
#            optical_HICP = pd.read_csv(
#                "../data/HICP/HICP-Corrective-eye-glasses-and-contact-lenses-France-Annual-parts-per-1000.csv"
#            )
#            optical_expenditures[key] = adjusted_price(optical_HICP, sum, i + 2014)
#            # print("test")
#        else:
#            optical_expenditures[key] = sum
#
#    return optical_expenditures


def gov_exp(inflation_adjustment: bool, sector: str, mask: Dict[str, List[str]]): #but this is exclusive mask, I have to do sth that can put a OR condition on the masks
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

    elements = os.listdir("../Open-LPP-data/base_complete")
    nb_LPP_codes = {}
    for i in range(int(len(elements) / 2)):
        df = pd.read_csv(
            f"../Open-LPP-data/base_complete/OPEN_LPP_{(i+2014)}.CSV",
            encoding="ISO-8859-1",
            sep=";",
        )
        if mask == {}:
            pass
        else:
            for filter in mask:
                if (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC1"):
                    final_mask = df[mask[filter][1]] == filter
                elif (mask[filter][0] == "equality") & (mask[filter][1] == "L_SC2"):
                    final_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "equality") & (mask[filter][1] == "L_CODE_LPP"):
                    final_mask = df[mask[filter][1]] == filter

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC1"):
                    final_mask = df[mask[filter][1]].str.contains(filter)

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_SC2"):
                    final_mask = df[mask[filter][1]].str.contains(filter)

                elif (mask[filter][0] == "contains") & (mask[filter][1] == "L_CODE_LPP"):
                    final_mask = df[mask[filter][1]].str.contains(filter)

                print(i + 2014)
                df = df[final_mask]

        df = pd.DataFrame(
            {
                "L_SC1": df["L_SC1"],
                "L_SC2": df["L_SC2"],
                "CODE_LPP": df["CODE_LPP"],
                "Quantity": df["QTE"],
                "Financing": df["REM"],
            }
        )
        df.reset_index(inplace=True)
        df.drop(columns="index", inplace=True)
        df["Total"] = df["Quantity"] * df["Financing"]
        sum = df["Total"].sum()
        key = str(i + 2014)
        nb_LPP_codes[key] = len(df)

        if inflation_adjustment == True:
            if sector == "optical":
                optical_HICP = pd.read_csv(
                    "../data/HICP/HICP-Corrective-eye-glasses-and-contact-lenses-France-Annual-parts-per-1000.csv"
                )
                expenditures[key] = adjusted_price(optical_HICP, sum, i + 2014)
            elif sector == "hearing":
                print(
                    "Hearing HICP is neagligeable face to the amount of money and the trend is the same whether we take HICP in count or not."
                )
            elif sector == "all":
                pass
        else:
            expenditures[key] = sum

    return expenditures, nb_LPP_codes


# example : test = gov_exp(inflation_adjustment=False, sector="optical", mask={"LUNETTES":["equality","L_SC2"]})