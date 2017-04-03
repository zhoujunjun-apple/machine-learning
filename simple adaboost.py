import numpy as np
import pandas as pd
import operator as opt
import csv

data = {
    'feature': range(0, 10),
    'output': pd.Series([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]),
    'weight': pd.Series(np.repeat(0.1, 10))
}

class adaboost:
    def __init__(self, error, max_iter):
        self.train_data = pd.DataFrame(data)  # 训练数据
        self.max_iter = max_iter  # 最大迭代次数
        self.error = error  # 所希望达到的错误率
        self.train_len = len(self.train_data)  # 样本个数
        self.final_model = pd.DataFrame(columns=('split', 'up', 'weight'))  # 保存弱分类器
        self.split = self.split_point()  # 训练数据特征的候选切分点

    # 取特征的相邻值平均值作为候选切分点
    def split_point(self):
        result = []
        for i in range(0, self.train_len - 1):
            tmp = (self.train_data.feature[i] + self.train_data.feature[i+1]) / 2.0
            result.append(tmp)
        return result

    # 计算每个切分点对应的误分类率
    # 每个切分点存在两种切分方向：大于切分点的样本取做正类 或者 小于切分点的样本取做正类
    # 两种切分方向对应的弱分类器误分类率之和总是为1，只需计算一种切分方向的弱分类器的误分类率
    def prediction(self):
        pred_wrong_error = []
        # 默认所有样本分作正类
        result_up = np.repeat(1, self.train_len)
        # 外循环遍历每一个候选的切分点
        for i in range(0, self.train_len - 1):
            tmp = 0
            # 内循环遍历每一个分类样本
            for j in range(0, self.train_len):
                # 小于切分点的样本才分作负类
                if self.train_data.feature[j] < self.split[i]:
                    result_up[j] = -1
                # 遍历样本时同时累积对应切分点的误分类率
                if result_up[j] != self.train_data.output[j]:
                    tmp += self.train_data.weight[j]
            pred_wrong_error.append(tmp)
        return pred_wrong_error

    # 保存最佳弱分类器：切分点，切分方向，分类器权重
    # 切分方向定义为：大于切分点的样本分作正类，对应up==True；小于切分点的样本分作正类，对应up==False
    # 首先找到最佳的切分点，再根据误分类率确定切分方向
    # 最佳切分点的寻找依据为：该切分点处的两个弱分类器的误分类率相差最大 <==> |x-(1-x)|=|2x-1|最大
    # 确定切分方向：因为pred_wrong是在将大于切分点的样本分作正类的前提下计算的，
    # 若最佳切分点处的误分类率大于0.5，则应该取反方向的弱分类器，否则取当前方向的弱分类器
    def record_model(self, pred_wrong):
        # 寻找最佳切分点的索引
        best_split_index, best_split = max(enumerate(np.abs(np.dot(2, pred_wrong) - 1)), key=opt.itemgetter(1))
        # 找到对应的最佳切分点
        best_split = self.split[best_split_index]
        # 误分类率大于0.5代表应该取反方向的弱分类器，并求出对应的误分类率
        if pred_wrong[best_split_index] >= 0.5 :
            up = False
            true_pred = 1 - pred_wrong[best_split_index]
        else:
            up = True
            true_pred = pred_wrong[best_split_index]

        # 根据误分类率计算弱分类器权重
        model_weight = 0.5 * np.log(1.0/true_pred - 1)
        recent_model_index = len(self.final_model)
        # 保存弱分类器的参数
        self.final_model.loc[recent_model_index] = [best_split, up, model_weight]

    # 根据最近一个弱分类器的分类结果更新样本的权重
    def train_reweight(self):
        # 找到最近的弱分类器
        recent_model_index = len(self.final_model)
        best_split, up, model_weight = self.final_model.loc[recent_model_index-1]
        # 根据up值确定每个样本对应的预测类别
        if up :
            for i in range(0, self.train_len):
                weight_old = self.train_data.weight[i].copy()
                output_old = self.train_data.output[i].copy()
                if self.train_data.feature[i] < best_split:
                    self.train_data.weight[i] = weight_old * np.exp(-model_weight * -output_old)
                else:
                    self.train_data.weight[i] = weight_old * np.exp(-model_weight * output_old)
        else :
            for i in range(0, self.train_len):
                weight_old = self.train_data.weight[i].copy()
                output_old = self.train_data.output[i].copy()
                if self.train_data.feature[i] < best_split:
                    self.train_data.weight[i] = weight_old * np.exp(-model_weight * output_old)
                else:
                    self.train_data.weight[i] = weight_old * np.exp(-model_weight * -output_old)
        # 归一化样本的权重分布
        self.train_data.weight = np.divide(self.train_data.weight, np.sum(self.train_data.weight))

    # 计算强分类器在input上的预测结果
    def final_classifier(self, input):
        len_input = len(input)  # 待分类的样本个数
        len_model = len(self.final_model)  # 弱分类器的个数
        result = []
        # 外循环遍历每一个待分类的样本
        for i in range(0, len_input):
            tmp_result = 0
            # 内循环遍历每个弱分类器对样本的类别进行预测
            for j in range(0, len_model):
                best_split, up, model_weight = self.final_model.loc[j]
                if up:
                    if input[i] < best_split:
                        tmp_result += -1 * model_weight
                    else:
                        tmp_result += 1 * model_weight
                else:
                    if input[i] < best_split:
                        tmp_result += 1 * model_weight
                    else:
                        tmp_result += -1 * model_weight
            result.append(np.sign(tmp_result))
        return result

    # 计算当前的强分类器的分类错误率
    def train_error(self, pred_result):
        error_result = np.sum(np.not_equal(pred_result, self.train_data.output)) / float(self.train_len)
        return error_result

    # 保存弱分类器参数到csv文件
    def save_model(self):
        csvfile = file('adaboost_output.csv','wb')
        writer = csv.writer(csvfile)
        writer.writerow(['split_point','up','model_weight'])
        for i in range(0, len(self.final_model)):
            writer.writerow(self.final_model.loc[i])
        csvfile.close()

    # 主入口程序
    def Ada(self):
        for iter in range(0, self.max_iter):
            pred_wrong = self.prediction()
            self.record_model(pred_wrong)
            self.train_reweight()
            recent_result = self.final_classifier(self.train_data.feature)

            if self.train_error(recent_result) < self.error:
                self.save_model()
                break
        self.save_model()

ada_model = adaboost(0.01, 5)
ada_model.Ada()
