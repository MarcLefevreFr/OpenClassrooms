import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


# lors des réutilisation, attention à LEN(GROUPS)

# anova(df, "var_qual", "var_quant")
def anova(df, vqual, vquant):
    
    #df = df[(df[vqual].notnull()) & (df[vquant]).notnull()]
    val_qual = []
    val_qual = df[vqual].value_counts().index
    groups =[df[df[vqual] == val][vquant] for val in val_qual]
    
    print(f"TABLE ANOVA {vqual} en fonction {vquant}")
    print()
    
    # # test normalité shapiro  
    # test_norm = True
    
    # for u in groups :
    #     if st.shapiro(u)[1] < 0.05 :
    #         test_norm = False
        
    # if test_norm == True:
    #     print("Test de normalité de Shapiro-Wilk : OK")
    # else :
    #     print("Test de normalité de Shapiro-Wilk : RATE")
        
    # # test homogénéité de Bartlett
    
    # if len(groups) == 5 :
    #     bart_p = st.bartlett(groups[0], groups[1], groups[2], groups[3], groups[4])[1]
    #     if bart_p < 0.05 :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test RATE")
    #     else :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test OK")
    #     print()   

    # if len(groups) == 4 :
    #     bart_p = st.bartlett(groups[0], groups[1], groups[2], groups[3])[1]
    #     if bart_p < 0.05 :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test RATE")
    #     else :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test OK")
    #     print()

    
    # calcul table ANOVA
    lin_mod = ols("Q(vquant) ~ Q(vqual)", data = df).fit()
    t_anova = sm.stats.anova_lm(lin_mod)
    
    print(t_anova)
    
    # calcul SCT
    SCT = t_anova["sum_sq"].sum()
    str_SCT = "{:2e}".format(SCT)
    print(f"SCT                         "+str_SCT)
    
    #calcul R2
    R2 = t_anova["sum_sq"][0] / SCT
    print(f"R2 = {R2}")


def anov(df, vqual, vquant):
    
    #df = df[(df[vqual].notnull()) & (df[vquant]).notnull()]
    val_qual = []
    val_qual = df[vqual].value_counts().index
    groups =[df[df[vqual] == val][vquant] for val in val_qual]
    
    #print(f"TABLE ANOVA {vqual} en fonction {vquant}")
    #print()
    
    # # test normalité shapiro  
    # test_norm = True
    
    # for u in groups :
    #     if st.shapiro(u)[1] < 0.05 :
    #         test_norm = False
        
    # if test_norm == True:
    #     print("Test de normalité de Shapiro-Wilk : OK")
    # else :
    #     print("Test de normalité de Shapiro-Wilk : RATE")
        
    # # test homogénéité de Bartlett
    
    # if len(groups) == 5 :
    #     bart_p = st.bartlett(groups[0], groups[1], groups[2], groups[3], groups[4])[1]
    #     if bart_p < 0.05 :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test RATE")
    #     else :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test OK")
    #     print()   

    # if len(groups) == 4 :
    #     bart_p = st.bartlett(groups[0], groups[1], groups[2], groups[3])[1]
    #     if bart_p < 0.05 :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test RATE")
    #     else :
    #         print(f"Test d'homogénéité de Bartlett, p = {bart_p}, test OK")
    #     print()

    
    # calcul table ANOVA
    lin_mod = ols("Q(vquant) ~ Q(vqual)", data = df).fit()
    t_anova = sm.stats.anova_lm(lin_mod)
    
    #print(t_anova)
    
    # calcul SCT
    SCT = t_anova["sum_sq"].sum()
    str_SCT = "{:2e}".format(SCT)
    #print(f"SCT                         "+str_SCT)
    
    #calcul R2
    R2 = t_anova["sum_sq"][0] / SCT
    #print(f"R2 = {R2}")
    
    return vquant, vqual, R2


def best_features_level(n, df, liste, target, nb_rep) :
    
    # pour une liste de VQuant rapport à une cible donnée, calcule les anova/R2 en fonctin du Top-N de valeurs
    # gardées dans les catégories, et ce pour N décroissant de sa val initiale "n" à 1.
    # ca peut permettre de trouver le level optimal du nb de caté dans des var catégorielles
    
    resultats = []
    
    for i in range(n, 0, -1):
        
        for col in liste :
            
            df1 = df.copy()
            
            top = df1[col].value_counts().index.tolist()[:i+1]

            labels = df1[col].value_counts().index.tolist()

            new = [l if l in top else "Other" for l in labels]

            dico = {k:v for k, v in zip(labels, new)}

            df1[col] = df1[col].map(dico)

            res = anov(df1, col, target)
            
            resultats.append(np.array([i, res[0], res[1], res[2]]))
        
    df_res = pd.DataFrame(data = np.array(resultats), columns = ["i", "vquant", "vqual", "r2"])
    
    df_res['i'] = df_res['i'].astype("int")
    df_res['r2'] = df_res['r2'].astype("float")
    
    best_res = []
    
    for col in liste :
        
        df_res2 = df_res[["vqual", "i", "r2"]][df_res["vqual"] == col].sort_values("r2", ascending = False).head(nb_rep).copy()
        
        print(len(best_res))
        print(f"Pour la variable : {col} - Et la cible : {target}")
        print(f"dont le feature level original est : {df[col].unique().shape[0]}")
        print()
        print(df_res2)
        print()
        print("------------------------------------")
        print()
        
        best_res.append([df_res2.values[0][0], df_res2.values[0][1], df_res2.values[0][2]])
        
    return best_res



def reduce_vcat_to_nval(n, df, vqual, newname = None):
    
            top = df[vqual].value_counts().index.tolist()[n]

            labels = df[vqual].value_counts().index.tolist()

            new = [l if l in top else "Other" for l in labels]

            dico = {k:v for k, v in zip(labels, new)}
            
            if newname :
                
                df[newname] = df[vqual].map(dico)
            
            else :
                
                new_name = f"{vqual}_red2_{n}"
                df[new_name] = df[vqual].map(dico)


def anova_g(df, liste, target, ret_resultats = True, imprime = False, graph = False):
    
    # génère boxplot et anova pour une df donnée, une cibles QUANT, et une liste de var QUAL
    
    dft = df.copy()    
    
    meanp = {"marker":"s", "markerfacecolor":"white", "markeredgecolor":"white"}

    resultats = []
    
    for col in liste :
        
        dft[col] = dft[col].astype("str") # convertit la var caté en string si ce n'est le cas

        if graph :

            plt.figure(figsize=(8,3))
            plt.title(f"{target} en fonction de {col}")
            sns.boxplot(dft[col], dft[target].sort_values(), showmeans = True, meanprops = meanp)
            plt.show()
        

        lin_mod = ols("Q(target) ~ Q(col)", data = df).fit()
        t_anova = sm.stats.anova_lm(lin_mod)

        # calcul SCT
        SCT = t_anova["sum_sq"].sum()
        str_SCT = "{:2e}".format(SCT)

        #calcul R2
        R2 = t_anova["sum_sq"][0] / SCT

        res = [target, col, R2, t_anova.loc["Q(col)"]["PR(>F)"]]
        resultats.append(res)

        
        if imprime :

            print(f"SCT                         "+str_SCT)
            print(t_anova)
            print(f"R2 = {R2}")
            print()

    if ret_resultats :

        df_res = pd.DataFrame(data = np.array(resultats), columns = ["vquant", "vqual", "R2", "p-value"])
        df_res = df_res.sort_values("R2", ascending = False)

        return df_res





def anova_d(df, liste, targets):
    
    # génère anova pour une df donnée, une cible QUANT, et une liste de var QUAL
    
    dft = df.copy()
    
    for tar in targets :
    
        for var in liste :
            
            if var != tar :

                dft[var] = dft[var].astype("str") # convertit la var caté en string si ce n'est le cas

                plt.title(f"{tar} en fonction de {var}")
                sns.boxplot(dft[var], dft[tar].sort_values(), showmeans = True, meanprops = meanp)
                plt.show()
                anova(dft, tar, var)
                print()


# INUTILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def test_chi2(df, liste, target):
    
    results = []
    dft = df.copy()
    
    for col in liste + [target] :
        
        if dft[col].dtypes in ["float64", "int64"] :
            
            dft[col] = dft[col].astype("str")
    
    for var in liste :

        cont_tab = df[[var, target]].pivot_table(index=var, columns=target, aggfunc=len, margins=True, 
                                  margins_name="Total").fillna(0).copy().astype(int)

        indicateurs = [var, target] + list(st.chi2_contingency(cont_tab))

        results.append(indicateurs)

        print()                
        print(f"Test Chi2 de {target} en fonction de {var}")
        print(f"Val CHI2 : {indicateurs[2]}")
        print(f"p-value  : {indicateurs[3]}")
        print(f"dof      : {indicateurs[4]}")
        print()
        print("------------------------------------------")


def test_chi2_g(df, liste, target, imprime = False, barplot = False, heatmap = False):
    
    results = []
    dft = df.copy()
    
    if dft[target].dtypes in ["float64", "int64"] :
            
            dft[target] = dft[target].astype("str") 
    
    for col in liste :
        
        if dft[col].dtypes in ["float64", "int64"] :
            
            dft[col] = dft[col].astype("str")
    
    for col in liste :

        if col != target :

            cont_tab = dft[[target, col]].pivot_table(index=target, columns=col, aggfunc=len, margins=True, 
                                      margins_name="Total").fillna(0).copy().astype(int)
            
            indicateurs = [col, target] + list(st.chi2_contingency(cont_tab))

            results.append(indicateurs[:-1])
            
            # stacked barplot
            if barplot :            
                
                data_graph = pd.DataFrame({
                k : [cont_tab.loc[k][i] for i in dft[col].unique()] for k in dft[target].unique()
                }, index = dft[col].unique())
                data_graph.plot(kind = "bar", stacked = True)
                plt.show()
            
            # Heatmap
            if heatmap :
                tx = cont_tab.loc[:,["Total"]]
                ty = cont_tab.loc[["Total"],:]
                n = len(df)
                indep = tx.dot(ty) / n
                c = cont_tab.fillna(0)
                measure = (c-indep)**2/indep
                xi_n = measure.sum().sum()
                table = measure/xi_n
                ax = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])

                # code servant à avoir des "bordures propres"...
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.7, top - 0.5)
                left, right = ax.get_xlim()
                ax.set_xlim(left - 0.5, right + 0.5)

                plt.show()
            
            if imprime :

                print(f"Test Chi2 de target = {target} en fonction de var = {col}")
                print()                
                print(f"Val CHI2 : {indicateurs[2]}")
                print(f"p-value  : {indicateurs[3]}")
                print(f"dof      : {indicateurs[4]}")
                print()
                print("------------------------------------------")

    df_res = pd.DataFrame(data = np.array(results), columns = ["var_liste", "cible", "Chi2", "p-value", "dof"])
    df_res = df_res.sort_values("Chi2", ascending = False) 
        
    return df_res


def table_cor(df, liste_var_quant) :

    # affiche la table de corrélation d'une liste de variables quantitatives

    tab = np.abs(df[liste_var_quant].corr())

    mask = np.triu(tab)

    plt.figure(figsize = (12,12))
    ax = sns.heatmap(tab, annot = True, fmt = ".1f", vmin = 0, vmax = 1, center = 0.5, cmap= 'coolwarm', 
                linecolor = "white", linewidth = 0.5, square = True, mask = mask)

    # les 4 lignes suivantes servent à corriger des problèmes de marge du graph...
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.7, top - 0.5)
    left, right = ax.get_xlim()
    ax.set_xlim(left - 0.5, right + 0.5)

    plt.show()

                










































