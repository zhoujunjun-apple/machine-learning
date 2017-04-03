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
        self.train_data = pd.DataFrame(data)  # training data
        self.max_iter = max_iter  # the maximum iteration
        self.error = error  # the training error we would like to achive
        self.train_len = len(self.train_data)  # sample number
        self.final_model = pd.DataFrame(columns=('split', 'up', 'weight'))  # for recording the basic classifier
        self.split = self.split_point()  # the candidate split points
        
    # finding candidate split point
    def split_point(self):
        result = []
        for i in range(0, self.train_len - 1):
            tmp = (self.train_data.feature[i] + self.train_data.feature[i+1]) / 2.0
            result.append(tmp)
        return result

    # calculate the error rate for each split point
    # there is two splitt "directions" for each split point: positive class for x > split_point or negative class for x < split_point
    # ONLY need to calculate one "direction" because the summation of those two error rate always equals to ONE.
    def prediction(self):
        pred_wrong_error = []
        # all the samples are positive class for default
        result_up = np.repeat(1, self.train_len)
        # walk through every split point
        for i in range(0, self.train_len - 1):
            tmp = 0
            # walk through every sample for each split point
            for j in range(0, self.train_len):
                # sample is negative ONLY less than split point value
                if self.train_data.feature[j] < self.split[i]:
                    result_up[j] = -1
                # accumulate error rate for each split point
                if result_up[j] != self.train_data.output[j]:
                    tmp += self.train_data.weight[j]
            pred_wrong_error.append(tmp)
        return pred_wrong_error

    # record the best basic classifier: split point, split "direction" and classifier's weight
    # split "direction" is up, which means up==True, only when samples are positive if it bigger than split point
    # the BEST split point can be found by finding the maximum value of |2x-1| in which x represents error rate of specific split point
    # the BEST split "direction" can be found by confirming whether x is less than 0.5 or not
    def record_model(self, pred_wrong):
        # find the index of best split point
        best_split_index, best_split = max(enumerate(np.abs(np.dot(2, pred_wrong) - 1)), key=opt.itemgetter(1))
        # fetch the best split value
        best_split = self.split[best_split_index]
        # if the error rate bigger than 0.5, than change the split "direction" and recalculate the true error rate
        if pred_wrong[best_split_index] >= 0.5 :
            up = False
            true_pred = 1 - pred_wrong[best_split_index]
        else:
            up = True
            true_pred = pred_wrong[best_split_index]

        # calculate the weight of basic classifier that we just find out
        model_weight = 0.5 * np.log(1.0/true_pred - 1)
        recent_model_index = len(self.final_model)
        # record the basic classifier
        self.final_model.loc[recent_model_index] = [best_split, up, model_weight]

    # refresh samples' weight according to the output of the latest basic classifier
    def train_reweight(self):
        # find and fetch the latest basic classifier
        recent_model_index = len(self.final_model)
        best_split, up, model_weight = self.final_model.loc[recent_model_index-1]
        # recalcuate the output of the latest basic classifier and record the new weight of samples
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
        # normalize and update the weight
        self.train_data.weight = np.divide(self.train_data.weight, np.sum(self.train_data.weight))

    # calculate the output of the final classifier
    def final_classifier(self, input):
        len_input = len(input) 
        len_model = len(self.final_model)  # fetch the number of basic classifier
        result = []
        
        for i in range(0, len_input):
            tmp_result = 0
            # walk through every basic classifier for each input sample, and predict the class
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

    # calculate the error rate of final classifier
    def train_error(self, pred_result):
        error_result = np.sum(np.not_equal(pred_result, self.train_data.output)) / float(self.train_len)
        return error_result

    # save all basic classifier to CSV file
    def save_model(self):
        csvfile = file('adaboost_output.csv','wb')
        writer = csv.writer(csvfile)
        writer.writerow(['split_point','up','model_weight'])
        for i in range(0, len(self.final_model)):
            writer.writerow(self.final_model.loc[i])
        csvfile.close()

    # the actual 'main' function
    def Ada(self):
        for iter in range(0, self.max_iter):
            pred_wrong = self.prediction()
            self.record_model(pred_wrong)
            self.train_reweight()
            recent_result = self.final_classifier(self.train_data.feature)

            if self.train_error(recent_result) < self.error:
                break
        self.save_model()

ada_model = adaboost(0.01, 5)
ada_model.Ada()
