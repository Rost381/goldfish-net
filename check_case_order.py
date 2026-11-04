# dump_algo.py   запуск: python dump_algo.py ESPORTSUSDT
import json, sys
from dotenv import load_dotenv
import place_limit_grid as g

load_dotenv()
ex = g.build_exchange("binance", "demo")
ex.load_markets()
sym = sys.argv[1].upper()
print("algo:", json.dumps(ex.fapiPrivateGetOpenAlgoOrders({"symbol": sym}), indent=2, ensure_ascii=False))
print("regular:", json.dumps(ex.fapiPrivateGetOpenOrders({"symbol": sym}), indent=2, ensure_ascii=False))