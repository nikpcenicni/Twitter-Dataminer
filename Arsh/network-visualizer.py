from jaal import Jaal
from jaal.datasets import load_got
import pandas as pd
# load the data
edge_df = pd.read_csv('followers.csv')
node_df = pd.read_csv('skipped.csv')
#edge_df, node_df = load_got()
# init Jaal and run server
Jaal(edge_df).plot()