import pandas as pd
import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import crm_data_prep, check_df, create_invoice_product_df

pd.set_option('display.max_columns', None)

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011",
                    engine="openpyxl")
df = df_.copy()
df.info()

df_prep = crm_data_prep(df)
df_prep.head()
check_df(df_prep)


def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (today_date - date.max()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    rfm.columns = ['recency', 'T', 'frequency', 'monetary']

    # CALCULATION OF MONETARY AVG & ADDING RFM DF
    temp_df = dataframe.groupby(["Customer ID", "Invoice"]).agg({"TotalPrice": ["mean"]})
    temp_df = temp_df.reset_index()
    temp_df.columns = temp_df.columns.droplevel(0)
    temp_df.columns = ["Customer ID", "Invoice", "total_price_mean"]
    temp_df2 = temp_df.groupby(["Customer ID"], as_index=False).agg({"total_price_mean": ["mean"]})
    temp_df2.columns = temp_df2.columns.droplevel(0)
    temp_df2.columns = ["Customer ID", "monetary_avg"]

    rfm = rfm.merge(temp_df2, how="left", on="Customer ID")
    rfm.set_index("Customer ID", inplace=True)
    rfm.index = rfm.index.astype(int)

    # CALCULATION OF WEEKLY RECENCY VE WEEKLY T FOR BGNBD
    rfm["recency_weekly"] = rfm["recency"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    # CONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]
    rfm["frequency"] = rfm["frequency"].astype(int)

    # BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly'],
            rfm['T_weekly'])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly'],
                                           rfm['T_weekly'])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly'],
                                           rfm['T_weekly'])

    # expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])
    # 6 MONTHS cltv_p
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # cltv_p_segment
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    rfm = rfm[["monetary_avg", "T", "recency_weekly", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]

    return rfm


cltv_p = create_cltv_p(df_prep)
check_df(cltv_p)
cltv_p.head()

cltv_p.groupby("cltv_p_segment").agg({"count", "mean"})

a_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "A"].index
b_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "B"].index
c_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "C"].index

# reduction of df's according to these ids
a_segment_df = df_prep[df_prep["Customer ID"].isin(a_segment_ids)]
b_segment_df = df_prep[df_prep["Customer ID"].isin(b_segment_ids)]
c_segment_df = df_prep[df_prep["Customer ID"].isin(c_segment_ids)]


def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True, low_memory=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True, low_memory=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules
# warning = takes too long time
for country in df.Country.unique():
    rules_a = create_rules(a_segment_df, country)
# DISCARD FROM FROOZEN SET
product_a = int(rules_a["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# warning = takes too long time
for country in df.Country.unique():
    rules_b = create_rules(b_segment_df, country)
# DISCARD FROM FROOZEN SET
product_b = int(rules_b["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# warning = takes too long time
for country in df.Country.unique():
    rules_c = create_rules(c_segment_df, country)
# DISCARD FROM FROOZEN SET
product_c = int(rules_c["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])


def check_id(stock_code):
    product_name = df_prep[df_prep["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return print(product_name)


check_id(20719)


cltv_p["recommended_product"] = ""

def id_get(dataframe, country):
    ids = dataframe[dataframe["Country"] == country]["Customer ID"].drop_duplicates()
    cltv_p.loc[(cltv_p.index.isin(ids)) & (cltv_p["cltv_p_segment"] == "A"), "recommended_product"] = product_a
    cltv_p.loc[(cltv_p.index.isin(ids)) & (cltv_p["cltv_p_segment"] == "B"), "recommended_product"] = product_b
    cltv_p.loc[(cltv_p.index.isin(ids)) & (cltv_p["cltv_p_segment"] == "C"), "recommended_product"] = product_c


for col in df.Country.unique():
    id_get(df_prep,col)

df_prep



