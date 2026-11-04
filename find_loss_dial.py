import pandas as pd
f = pd.read_csv("fills.csv")
o = f.groupby(["symbol", "orderId"]).agg(
    realized=("realizedPnl", "sum"), fee=("fee_usdt", "sum"), time=("time", "min")
).reset_index()
o["net"] = o.realized - o.fee
print(o[o.realized != 0].sort_values("net").head(5))