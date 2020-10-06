import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import scipy.stats as st

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




