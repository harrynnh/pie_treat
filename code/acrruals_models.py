import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare

os.chdir("/Users/harrynnh/workspace/misc/treat")

# Prepare data for analysis
ff12 = pd.read_csv("data/external/fama_french_12_industries.csv")
ff48 = pd.read_csv("data/external/fama_french_48_industries.csv")
us_base_sample = pd.read_feather("data/pulled/cstat_us_sample.feather")
us_base_sample = (
    us_base_sample.sort_values(["gvkey", "datadate"])
    .assign(
        sic=np.where(
            ~us_base_sample["sich"].isna(),
            us_base_sample["sich"].astype(str).str[0:4],
            us_base_sample["sic"],
        ).astype("float")
    )
    .loc[
        lambda x: (x["indfmt"] == "INDL")
        & (x["fic"] == "USA")
        & (~x["at"].isna())
        & (x["at"] > 0)
        & (x["sale"] > 0)
        & (~x["sic"].isna())
        & ((x["sic"] < 6000) | (x["sic"] > 6999))
    ]
    .merge(ff48, how="left", on="sic")
    .merge(ff12, how="left", on="sic")
    .loc[lambda x: ~x["ff48_ind"].isna() & ~x["ff12_ind"].isna()]
)
duplicated = us_base_sample.loc[us_base_sample.duplicated(subset=["gvkey", "fyear"])]
if len(duplicated) == 0:
    print("No duplicated rows found")
else:
    sys.exit("There are dublicated firm-year observations")

# Estimate accruals models
def mj_mod(df):
    mod = "tacc ~ inverse_a + drev + ppe"
    mj_res = smf.ols(mod, data=df).fit()
    df["mj_da"] = mj_res.resid
    df["mj_adjr2"] = mj_res.rsquared
    df["mj_nobs"] = mj_res.nobs
    return df


def dd_mod(df):
    mod = "dwc ~ cfo_lag + cfo + cfo_lead"
    dd_res = smf.ols(mod, data=df).fit()
    df["dd_da"] = dd_res.resid
    df["dd_adjr2"] = dd_res.rsquared
    df["dd_nobs"] = dd_res.nobs
    return df


# Calculate modified Jones model accruals and statistics
# Methodology is somewhat loosely based on Hribar and Nichols (JAR, 2007)
# https://doi.org/10.1111/j.1475-679X.2007.00259.x
shift_cols = ["at", "sale"]
min_obs = 10
us_base_sample = us_base_sample.assign(
    at_lag=lambda x: x.groupby("gvkey")["at"].shift(1),
    sale_lag=lambda x: x.groupby("gvkey")["sale"].shift(1),
    at_lead=lambda x: x.groupby("gvkey")["at"].shift(-1),
    sale_lead=lambda x: x.groupby("gvkey")["sale"].shift(-1),
    tacc=lambda x: (x["ibc"] - x["oancf"]) / x["at_lag"],
    drev=lambda x: (x["sale"] - x["sale_lag"] + x["recch"]) / x["at_lag"],
    inverse_a=lambda x: 1 / x["at_lag"],
    ppe=lambda x: x["ppegt"] / x["at_lag"],
    avgta=lambda x: (x["at"] + x["at_lag"]) / 2,
    cfo=lambda x: x["oancf"] / x["avgta"],
    cfo_lag=lambda x: x.groupby("gvkey")["cfo"].shift(1),
    cfo_lead=lambda x: x.groupby("gvkey")["cfo"].shift(-1),
    dwc=lambda x: -(x["recch"] + x["invch"] + x["apalch"] + x["aoloch"]) / x["avgta"],
)

mj_sample = (
    us_base_sample.loc[
        lambda x: (~x["tacc"].isna()) & (~x["drev"].isna()) & (~x["ppe"].isna()),
        ["gvkey", "ff48_ind", "fyear", "tacc", "drev", "inverse_a", "ppe"],
    ]
    .groupby(["ff48_ind", "fyear"])
    .filter(lambda x: len(x) >= min_obs)
)
mj_wcols = ["tacc", "drev", "inverse_a", "ppe"]
mj_sample.loc[:, mj_wcols] = mj_sample.loc[:, mj_wcols].clip(
    lower=mj_sample[mj_wcols].quantile(0.01),
    upper=mj_sample[mj_wcols].quantile(0.99),
    axis=1,
)  # specify cols in lower & upper doesn't matter
mj_sample = mj_sample.groupby(["ff48_ind", "fyear"]).apply(mj_mod)

dd_sample = (
    us_base_sample.loc[
        lambda x: (~x["dwc"].isna())
        & (~x["cfo"].isna())
        & (~x["cfo_lag"].isna())
        & (~x["cfo_lead"].isna()),
        ["gvkey", "ff48_ind", "fyear", "dwc", "cfo", "cfo_lag", "cfo_lead"],
    ]
    .groupby(["ff48_ind", "fyear"])
    .filter(lambda x: len(x) >= min_obs)
)
dd_wcols = ["dwc", "cfo", "cfo_lag", "cfo_lead"]
dd_sample.loc[:, dd_wcols] = dd_sample.loc[:, dd_wcols].clip(
    lower=dd_sample[dd_wcols].quantile(0.01),
    upper=dd_sample[dd_wcols].quantile(0.99),
    axis=1,
)
dd_sample = dd_sample.groupby(["ff48_ind", "fyear"]).apply(dd_mod)

selected_cols = [
    "gvkey",
    "conm",
    "fyear",
    "ff12_ind",
    "ff48_ind",
    "ta",
    "sales",
    "mktcap",
    "ln_ta",
    "ln_sales",
    "ln_mktcap",
    "mj_da",
    "dd_da",
    "mj_ada",
    "dd_ada",
    "mtb",
    "sales_growth",
    "leverage",
    "ppe_ta",
    "int_ta",
    "gwill_ta",
    "ceq_ta",
    "leverage",
    "acq_sales",
    "cogs_sales",
    "ebit_sales",
    "ebit_avgta",
    "cfo_avgta",
    "tacc_avgta",
]
smp = (
    us_base_sample.sort_values(["gvkey", "fyear"])
    .merge(mj_sample[["gvkey", "fyear", "mj_da"]], how="left", on=["gvkey", "fyear"])
    .merge(dd_sample[["gvkey", "fyear", "dd_da"]], how="left", on=["gvkey", "fyear"])
    .assign(
        mktcap=lambda x: x["csho"] * x["prcc_f"],
        ln_ta=lambda x: np.log(1 + x["at"]),
        ln_sales=lambda x: np.log(1 + x["sale"]),
        ln_mktcap=lambda x: np.log(1 + x["mktcap"]),
        mtb=lambda x: (x["csho"] * x["prcc_f"]) / x["ceq"],
        sales_growth=lambda x: (np.log(1 + x["sale"]) / np.log(1 + x["sale_lag"])),
        leverage=lambda x: x["lt"] / x["at"],
        ppe_ta=lambda x: x["ppent"] / x["at"],
        int_ta=lambda x: x["intan"] / x["at"],
        gwill_ta=lambda x: x["gdwl"] / x["at"],
        acq_sales=lambda x: (
            np.where(~x["aqs"].isna(), x["aqs"], 0)
            + np.where(~x["acqsc"].isna(), x["acqsc"], 0)
        )
        / x["sale"],
        cogs_sales=lambda x: x["cogs"] / x["sale"],
        ebit_sales=lambda x: (x["ib"] + x["xint"]) / x["sale"],
        ebit_avgta=lambda x: (x["ib"] + x["xint"]) / x["avgta"],
        cfo_avgta=lambda x: x["oancf"] / x["avgta"],
        tacc_avgta=lambda x: (x["ibc"] - x["oancf"]) / x["avgta"],
        ceq_ta=lambda x: x["ceq"] / x["at"],
        mj_ada=lambda x: np.abs(x["mj_da"]),
        dd_ada=lambda x: np.abs(x["dd_da"]),
    )
    .rename(columns={"at": "ta", "sale": "sales"})
    .loc[lambda x: ~x["mj_da"].isna(), selected_cols,]
)

# Regression analysis
da_cols = [
    "gvkey",
    "fyear",
    "mj_da",
    "dd_da",
    "ln_ta",
    "ln_mktcap",
    "mtb",
    "ebit_avgta",
    "sales_growth",
]
smp_da = smp.loc[:, da_cols,].replace([np.inf, -np.inf], np.nan)
# smp_da.isin([np.inf, -np.inf]).sum()
da_winz_cols = [
    "mj_da",
    "dd_da",
    "ln_ta",
    "ln_mktcap",
    "mtb",
    "ebit_avgta",
    "sales_growth",
]
smp_da.loc[:, da_winz_cols] = smp_da.groupby("fyear")[da_winz_cols].apply(
    lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99), axis=1)
)
smp_da = smp_da.dropna().set_index(["gvkey", "fyear"])


def mj_pointless_mod(df):
    mod = (
        "mj_da ~ ln_ta + mtb + ebit_avgta + sales_growth + EntityEffects + TimeEffects"
    )
    mj_res = PanelOLS.from_formula(mod, data=df).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    return mj_res


def dd_pointless_mod(df):
    mod = (
        "dd_da ~ ln_ta + mtb + ebit_avgta + sales_growth + EntityEffects + TimeEffects"
    )
    dd_res = PanelOLS.from_formula(mod, data=df).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    return dd_res


print(smp_da[da_winz_cols].describe().transpose())
corr_pearson = np.triu(smp_da[da_winz_cols].corr(method="pearson"), k=1)
corr_spearman = np.tril(smp_da[da_winz_cols].corr(method="spearman"), k=-1)
print(
    compare(
        {
            "Modified Jones": mj_pointless_mod(smp_da),
            "Dechow & Dichev": dd_pointless_mod(smp_da),
        }
    )
)
corr_full = corr_pearson + corr_spearman
np.fill_diagonal(corr_full, 1)
corr_df = pd.DataFrame(corr_full, columns=da_winz_cols, index=da_winz_cols,)
