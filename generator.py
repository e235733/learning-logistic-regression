import numpy as np
import random

class Generator:
    rng = np.random.default_rng(0)

    def generate_two_centroids(self, range, dist):
        """
        2つの重心を生成する。
        
        Parameters
        ----------
        range : int
            生成する重心座標の上限
        dist : int
            2つの重心の最小距離

        Returns
        ----------
        left_centroid : int
            座標の値が小さい方の重心
        right_centroid : int
            座標の値が大きい方の重心 
        """
        left_centroid = random.randint(0, range - dist)
        right_centroid = random.randint(left_centroid + dist, range)

        return left_centroid, right_centroid

    def generate_single_val_data(self, num_data, range, dist, sigma):
        """
        2つのクラスタからなるデータとラベルを生成する。dist <= range / 2 である必要がある。
        
        Parameters
        ----------
        num_data : int
            生成するデータの数
        range : int
            生成するデータの上限
        dist : int
            2つの重心の最小距離
        sigma : float
            重心まわりの分布における標準偏差

        Returns
        ----------
        explain : np.array
            0 から range の範囲内のデータで、データ数は num_data 
        depend : np.array
            0 か 1 の正解ラベル
        """
        # 重心の生成
        left_centroid, right_centroid = self.generate_two_centroids(range, dist)
        #print("left_centroid:", left_centroid, ", right_centroid:", right_centroid)

        # データ数を2つに割り振る
        num_left_explain = random.randint(1, num_data - 1)
        num_right_explain = num_data - num_left_explain
        #print("num_left_explain:", num_left_explain, ", num_right_explain:", num_right_explain)

        # 正規分布のデータを作成
        left_explain = self.rng.normal(left_centroid, sigma, num_left_explain)
        right_explain = self.rng.normal(right_centroid, sigma, num_right_explain)
        left_explain = np.clip(left_explain, 0, 100)
        right_explain = np.clip(right_explain, 0, 100)
        #print("left_explain:\n", left_explain, "\nright_explain:\n", right_explain)

        # 0 か 1 の目的変数を割り当てる
        random_bool = random.choice([True, False])
        left_depend = None
        right_depend = None
        if random_bool:
            left_depend = np.zeros(num_left_explain)
            right_depend = np.ones(num_right_explain)
        else:
            left_depend = np.ones(num_left_explain)
            right_depend = np.zeros(num_right_explain)
        #print("left_depend:\n", left_depend, "\nright_depend:\n", right_depend)
        
        # 結合してシャッフルする
        explain = np.hstack((left_explain, right_explain))
        depend = np.hstack((left_depend, right_depend))
        data = np.vstack((explain, depend)).T
        self.rng.shuffle(data)
        
        [explain, depend] = np.vsplit(data.T, 2)
        return explain, depend

if __name__ == "__main__":
    g = Generator()
    explain, depend = g.generate_single_val_data(100, 100, 30, 10)
    print("explain:\n", explain, "\ndepend\n", depend)