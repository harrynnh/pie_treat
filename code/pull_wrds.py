# This scrip is to download data for accurals models
import pandas as pd
import wrds
import os
os.chdir("/Users/harrynnh/workspace/misc/treat")
conn = wrds.Connection(wrds_username="harrynnh")

# Wrtie a function to check if file exits; true, add date-timet to file name.
# Filters and variables
dyn_vars = ["gvkey", "conm", "cik", "fyear", "datadate", "indfmt", "sich",
            "consol", "popsrc", "datafmt", "curcd", "curuscn", "fyr",
            "act", "ap", "aqc", "aqs", "acqsc", "at", "ceq", "che", "cogs",
            "csho", "dlc", "dp", "dpc", "dt", "dvpd", "exchg", "gdwl", "ib",
            "ibc", "intan", "invt", "lct", "lt", "ni", "capx", "oancf",
            "ivncf", "fincf", "oiadp", "pi", "ppent", "ppegt", "rectr",
            "sale", "seq", "txt", "xint", "xsga", "costat", "mkvalt", "prcc_f",
            "recch", "invch", "apalch", "txach", "aoloch",
            "gdwlip", "spi", "wdp", "rcp"]
dyn_vars_str = ", ".join(dyn_vars)
stat_vars = ["gvkey", "loc", "sic", "spcindcd", "ipodate", "fic"]
stat_vars_str = ", ".join(stat_vars)
cs_filter = "consol='C' and (indfmt='INDL' or indfmt='FS') and datafmt='STD' and popsrc='D'"

res = "SELECT " + dyn_vars_str + " FROM COMP.FUNDA" + " WHERE " + cs_filter
print("Pulling dynamic Compustat data from WRDS...")
wrds_us_dynamic = conn.raw_sql(res)
print("Done")
res2 = "SELECT " + stat_vars_str + " FROM COMP.COMPANY"
print("Pulling static Compustat data from WRDS...")
wrds_us_static = conn.raw_sql(res2)
print("Done")
conn.close()
wrds_us = pd.merge(wrds_us_dynamic, wrds_us_static, how="inner", on="gvkey")
wrds_us.to_feather("data/pulled/cstat_us_sample.feather")
