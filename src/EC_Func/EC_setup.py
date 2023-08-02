import numpy as np
import torch
from src.EC_Func.EC_Classifier_Ball_Tree import EpistemicClassifier
from src.EC_Func.epsilon_performance_plot import epsilon_performance_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# from test import TestUtils
from pyswarms.single.global_best import GlobalBestPSO


class EC():
    def __init__(self, cfg, TestUtils):
        self.TestUtils = TestUtils
        self.cfg = cfg
        self.net = TestUtils.net.to("cpu")
        self.layer_interest = cfg['EC']['layer_interest']
        self.metric = cfg['EC']['metric']
        self.p = cfg['EC']['p']
        self.distance = cfg['EC']['distance']
        self.EC_model = EpistemicClassifier(self.net, self.layer_interest, self.metric, self.p)
        self.train_dataloaders_list = TestUtils.train_dataloaders_list
        self.test_dataloaders_list = TestUtils.test_dataloaders_list
        self.EC_model_list = [self.EC_model for _ in range(len(self.test_dataloaders_list))]
        self.epsilon_list = np.logspace(-1, 0, 30)
        self.verbose = True
        self.saved = False


    def EC_fit(self):
        self.net.eval()
        # Use the CPU device
        device = torch.device("cpu")

        # overall_acc = []
        acc_ik = []
        frac_ik = []
        frac_imk = []
        frac_idk = []

        for j in range(len(self.train_dataloaders_list)):
            x_train_all = []
            y_train_all = []
            # Iterate over the training dataloader
            for i, data in enumerate(self.train_dataloaders_list[j]):
                x_train = data[1].to(device)
                y_train = data[2].to(device)
                x_train_all.append(x_train)
                y_train_all.append(y_train.numpy())
            # Concatenate all test data
            x_train_all = torch.cat(x_train_all, dim=0)
            # print("x_train_all:", x_train_all.shape)
            y_train_all = np.concatenate(y_train_all, axis=0)
            # print("y_train_all:", y_train_all)

            print("start_fit")
            self.EC_model_list[j].fit(x_train_all, y_train_all)
            print("end_fit")

            x_test_all = []
            y_test_all = []
            # Collect all test data
            for i, data in enumerate(self.test_dataloaders_list[j]):
                x_test = data[1].to(device)
                y_test = data[2].to(device)
                y_test_all.append(y_test)
                x_test_all.append(x_test)
            x_test_all = torch.cat(x_test_all, dim=0)
            y_test_all = torch.cat(y_test_all, dim=0).numpy()

            y_true = y_test_all
            # overall_acc.append([])
            acc_ik.append([])
            frac_ik.append([])
            frac_imk.append([])
            frac_idk.append([])
            for eps in self.epsilon_list:
                # y_pred = self.EC_model_list[j].predict_class(x_test_all, dist=[eps], mode = "epsilon_ball")
                y_pred = self.EC_model_list[j].predict_class(x_test_all, n_neigh = 30, dist=[eps], mode = "fusion")

                trusted_index = np.where(y_pred < (np.max(y_true) + 0.1))[0]
                idk_index = np.where(y_pred == (np.max(y_true).astype(np.int) + 1))[0]
                imk_index = np.where(y_pred == (np.max(y_true).astype(np.int) + 2))[0]
                # overall accuracy
                # overall_acc[j].append(accuracy_score(y_true, y_pred))
                # fraction of idk
                idk_l = len(idk_index) / len(y_pred)
                imk_l = len(imk_index) / len(y_pred)
                ik_l = len(trusted_index) / len(y_pred)

                frac_ik[j].append(ik_l)
                frac_imk[j].append(imk_l)
                frac_idk[j].append(idk_l)
                # print("trusted_index:", trusted_index, len(trusted_index))
                # print("y_true:", y_true)
                # print("y_pred", y_pred)

                if len(trusted_index) == 0:
                    acc_ik[j].append(0.0)
                else:
                    # print("y_true[trusted_index]:", y_true[trusted_index])
                    acc_ik[j].append(accuracy_score(y_true[trusted_index], y_pred[trusted_index]))

        acc_ik = np.mean(np.array(acc_ik), axis=0).tolist()
        frac_ik = np.mean(np.array(frac_ik), axis=0).tolist()
        frac_imk = np.mean(np.array(frac_imk), axis=0).tolist()
        frac_idk = np.mean(np.array(frac_idk), axis=0).tolist()

        fig, ax = plt.subplots(1, 1)
        ax.semilogx(self.epsilon_list, acc_ik, '--', linewidth=2)
        ax.semilogx(self.epsilon_list, frac_ik, '-', linewidth=2)
        ax.semilogx(self.epsilon_list, frac_imk, ':', linewidth=2)
        ax.semilogx(self.epsilon_list, frac_idk, '-.', linewidth=2)
        ax.legend(['Acc IK', 'Frac IK', 'Frac IMK', 'Frac IDK'])  # , 'Overall Acc'
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Fraction')
        plt.show()

    # stanley modify
    def EC_fit_multilayer(self):
        self.net.eval()
        self.train_dataloaders = self.TestUtils.train_dataloaders
        self.test_dataloaders = self.TestUtils.test_dataloaders

        # Assigning height and width from configuration
        self.h, self.w = self.cfg['data']['image_size']['h'], self.cfg['data']['image_size']['w']
        n_layers = len(self.layer_interest)  # number of layers

        # Use the CPU device
        device = torch.device("cpu")
        x_train_all = []
        y_train_all = []
        # Iterate over the training dataloader
        for i, data in enumerate(self.train_dataloaders):
            x_train = data[1].to(device)
            y_train = data[2].to(device)
            x_train_all.append(x_train)
            y_train_all.append(y_train.numpy())
        # Concatenate all test data
        x_train_all = torch.cat(x_train_all, dim=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        self.EC_model.fit(x_train_all, y_train_all)

        x_test_all = []
        y_test_all = []

        # Collect all test data
        for i, data in enumerate(self.test_dataloaders):
            x_test = data[1].to(device)
            y_test = data[2].to(device)
            y_test_all.append(y_test)
            x_test_all.append(x_test)

        # Concatenate all test data
        x_test_all = torch.cat(x_test_all, dim=0)
        y_test_all = torch.cat(y_test_all, dim=0)
        actual_pred = y_test_all.float().tolist()
        actual_avg_pred = y_test_all.float().mean().item()  # actual average prediction
        print("actual_value:", actual_pred)
        print("actual_avg_value:", actual_avg_pred)

        def ratio_func(distances, bounds):
            distances = distances.tolist()
            ratio_list = []
            for distance in distances:
                distance = np.clip(distance, bounds[0], bounds[1])
                pred = self.EC_model.predict_class(x_test_all, n_neigh = 30, dist=distance, mode = "fusion")
                # Correct classifications for classes <= 1 and for class 2
                correct_classifications_leq1 = sum((pred <= 1))
                correct_classifications = len(pred)

                # Handling the case where there are no correct classifications for class 2 to avoid division by zero
                if correct_classifications == 0:
                    ratio = 0  # You can define what the ratio should be when there are no correct classifications for class 2
                else:
                    ratio = correct_classifications_leq1 / correct_classifications

                ratio_list.append(-ratio)  # negative as we want to maximize the ratio, but PSO minimizes the function
            return ratio_list

        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # Call instance of PSO
        optimizer = GlobalBestPSO(n_particles=50, dimensions=n_layers, options=options)

        # Perform optimization
        bounds = (0, 1)
        cost, optimal_distance = optimizer.optimize(ratio_func, iters=300, verbose=3, bounds=bounds)

        optimal_pred = self.EC_model.predict_class(x_test_all, dist=optimal_distance.tolist())

        # Compute the optimal ratio
        correct_classifications_leq1 = sum((optimal_pred <= 1))
        correct_classifications = len(optimal_pred)
        optimal_ratio = correct_classifications_leq1 / correct_classifications if correct_classifications != 0 else 0

        print("Optimal distance:", optimal_distance)
        # print("Optimal corresponding normalized distance:", optimal_norm_dist)
        print("Optimal corresponding ratio:", optimal_ratio)

    def EC_validate_fit(self):
        self.train_dataloaders = self.TestUtils.train_dataloaders
        self.test_dataloaders = self.TestUtils.test_dataloaders
        self.net.eval()
        # Use the CPU device
        device = torch.device("cpu")
        predict_label_accuracy = []
        x_train_all = []
        y_train_all = []
        # Iterate over the training dataloader
        for i, data in enumerate(self.train_dataloaders):
            x_train = data[1].to(device)
            y_train = data[2].to(device)
            x_train_all.append(x_train)
            y_train_all.append(y_train.numpy())
            outputs = self.net(x_train)
            # print("outputs:", outputs.shape,"outputs value", outputs)
            predicted = torch.max(outputs, 1)[1]
            # print("predicted:", len(predicted), "y_train:", len(y_train))
            predict_label_accuracy_res = [1 if predicted[i] == y_train[i] else -1 for i in range(len(y_train))]
            predict_label_accuracy.extend(predict_label_accuracy_res)
        # print("len(predict_label_accuracy):", len(predict_label_accuracy))
        # Concatenate all test data
        x_train_all = torch.cat(x_train_all, dim=0)
        # print("x_train_all:", x_train_all.shape)
        y_train_all = np.concatenate(y_train_all, axis=0)
        # print("y_train_all:", y_train_all)

        print("start_fit")
        self.EC_model.fit(x_train_all, y_train_all)
        print("end_fit")

        x_test_all = []
        y_test_all = []
        if self.verbose:
            raw_outputs = torch.tensor([])
        # Collect all test data
        for i, data in enumerate(self.test_dataloaders):
            x_test = data[1].to(device)
            y_test = data[2].to(device)
            y_test_all.append(y_test)
            x_test_all.append(x_test)
            test_outputs = self.net(x_test)
            # print("outputs:", outputs.shape,"outputs value", outputs)
            if self.verbose:
                raw_outputs = torch.concat([raw_outputs, test_outputs], dim=0)
        x_test_all = torch.cat(x_test_all, dim=0)
        y_test_all = torch.cat(y_test_all, dim=0).numpy()
        if self.verbose:
            print("outputs:", raw_outputs.shape)
        y_true = y_test_all
        y_pred = self.EC_model.predict_class(x_test_all, n_neigh=30, dist=[0.5], mode="fusion")
        trusted_index = np.where(y_pred < (np.max(y_true) + 0.1))[0]
        idk_index = np.where(y_pred == (np.max(y_true).astype(np.int) + 1))[0]
        imk_index = np.where(y_pred == (np.max(y_true).astype(np.int) + 2))[0]
        # overall accuracy
        # overall_acc[j].append(accuracy_score(y_true, y_pred))
        # fraction of idk
        idk_l = len(idk_index) / len(y_pred)
        imk_l = len(imk_index) / len(y_pred)
        ik_l = len(trusted_index) / len(y_pred)
        print("*****************************************")
        print("idk:", idk_l, "imk_l:", imk_l, "ik:", ik_l)
        print("*****************************************")


        reliability = self.EC_model.predict_class_individual(x_test_all, n_neigh=30, dist=[0.5], mode="fusion", predict_label_accuracy = predict_label_accuracy)
        # print("reliability:", type(reliability))

        if self.saved:
            reliability_torch = torch.tensor(reliability)
            reliability_torch = torch.unsqueeze(reliability_torch, dim=1)
            # print("raw_outputs:", raw_outputs.shape, "reliability_torch:", reliability_torch.shape)
            saved_feature = torch.concat([raw_outputs, reliability_torch], dim=1)
            print(saved_feature.shape)

            import pandas as pd
            from openpyxl import Workbook
            numpy_array = saved_feature.detach().numpy()
            df = pd.DataFrame(numpy_array)
            file_path = 'rgb_feature.csv'  # Choose the desired file path
            df.to_csv(file_path, index=False)  # Set index=False if you don't want to save row numbers


        # print("start_fit")
        # self.EC_model = self.EC_model
        # self.EC_model.fit(x, y) # EC.fit(X_train, y_train_int)
        # print("end_fit")
        #
        # # pred = self.EC_model.predict_class(x_test, n_neigh=2)
        # # # pred = self.EC_model.predict_class(x_test, dist=self.distance)
        # # print("pred:", pred) # IMK: max+2, IDN: max+1,
        #
        # # plot the curve
        # fig, ax = plt.subplots(1, 1)
        # # epsilon_performance_curve(self.EC_model, self.net, x_test, y_test, epsilon_list=np.logspace(-4, 0.5, 30), plt=ax)
        # epsilon_performance_curve(self.EC_model, self.net, x_test, y_test, epsilon_list=np.logspace(-4, 10, 30), plt=ax)
        # plt.show()

