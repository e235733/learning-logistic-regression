from generator import Generator
from logic import Parameter
from plotter import Plotter

NUM_EXPLAIN = 1
NUM_DATA = 100
RANGE = 100
DIST = 30
SIGMA = 10
ETA_W = 1
ETA_B = 1
NUM_STEP = 20

INTERVAL = 0.2

# データの生成
generator = Generator()
explain, depend = generator.generate_single_val_data(NUM_DATA, RANGE, DIST, SIGMA)

# パラメータクラス
parameter = Parameter(explain, depend, NUM_EXPLAIN, ETA_B, ETA_W)
w = parameter.get_w()
b = parameter.get_b()

# 描画クラスをインスタンス化
plotter = Plotter(INTERVAL, RANGE, explain, depend)
plotter.show(w, b)

# 指定回数学習
for _ in range(NUM_STEP):
    parameter.shift()
    w = parameter.get_w()
    b = parameter.get_b()

    plotter.show(w, b)