from sklearn import tree
import DecisionTree_Structure as DTS
import sklearn as skl
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import os
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as AdaB
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV as GSCV
from collections import OrderedDict as OD
from sklearn import neural_network
from sklearn import random_projection
from sklearn import cluster
from sklearn import mixture
from sklearn import decomposition
from Data_load import *
import pandas as pd
from subprocess import call
from sklearn.pipeline import make_pipeline


class Learner:
    def __init__(self, estimator_name, param_dict, Data_dir, Fig_dir, dataset_name, cv=None, n_jobs=1,
                 USL_SL_Flag=False, train_sizes=np.linspace(.1, 1.0, 5), feature_extract=True, select_labels=[0, 1],
                 normalize_data=True, DoGridSearch=False, load_data=True):
        self.param_dict = param_dict
        self.estimator_name = estimator_name
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.USL_SL = USL_SL_Flag  # False implies USL, True implies SL
        self.ylim = [0.25, 1]
        self.cv = cv
        self.select_labels = select_labels
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes
        self.dataset_name = dataset_name
        self.Data_dir = Data_dir
        self.Fig_dir = Fig_dir
        self.feature_extract = feature_extract
        self.elapsed_time = 0
        self.DoGridSearch = DoGridSearch
        if load_data:
            self.Dataset_load(norm=normalize_data)
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        self.fig0 = fig
        self.ax0 = ax
        self.fig1 = fig1
        self.ax1 = ax1
        self.estimator_param_string = ''
        self.skip_train = 10
        self.figlc, self.axlc = plt.subplots()

    def cvgridsearch(self, skip_train=5):
        self.GS = GSCV(estimator=self.estimator, param_grid=self.param_dict, cv=self.cv, n_jobs=self.n_jobs,
                       return_train_score=True)
        X_train = self.X_train[0:-1:skip_train, :]
        y_train = self.y_train[0:-1:skip_train]
        self.GS.fit(X_train, y_train)
        # self.plot_results_cvgridsearch()
        self.table_results_cvgridsearch()

    def Dataset_load(self, norm=True):
        if (self.dataset_name == 'Abalone'):
            Abalone_data_filename = 'abalone.data'
            self.X_train, self.y_train, self.X_test, self.y_test = Load_Abalone(self.Data_dir,
                                                                                Data_filename=Abalone_data_filename)

        elif (self.dataset_name == 'Mnist'):
            Mnist_train_data_filename = 'mnist_train.csv'
            Mnist_test_data_filename = 'mnist_test.csv'
            self.X_train, self.y_train, self.X_test, self.y_test = Load_Mnist(Data_Dir, Mnist_train_data_filename,
                                                                              Mnist_test_data_filename,
                                                                              feature_select=self.feature_extract)
        elif (self.dataset_name == 'Cifar-10'):
            Cifar_data_dir = 'cifar-10-batches-py'
            Cifar_data_filename = 'data_batch_1'
            self.X_train, self.y_train, self.X_test, self.y_test = Load_Cifar(
                os.path.join(Data_Dir, Cifar_data_dir), select_labels=self.select_labels,
                feature_extract=self.feature_extract)
        if norm:
            self.norm_max = np.max(self.X_train)
            self.norm_mean = np.mean(self.X_train, axis=0)
            self.X_train = self.X_train / self.norm_max
            self.X_train = self.X_train - self.norm_mean
            self.X_test = self.X_test / self.norm_max
            self.X_test = self.X_test - self.norm_mean

    def plot_results_cvgridsearch(self):
        self.grid_search_result = dict()
        self.grid_search_result['mean_test_score'] = self.GS.cv_results_['mean_test_score']
        self.grid_search_result['mean_train_score'] = self.GS.cv_results_['mean_train_score']
        self.grid_search_result['mean_fit_time'] = self.GS.cv_results_['mean_fit_time']
        self.num_plot_axis = 0
        param_keylist = list(self.param_dict.keys())
        self.num_plot_axis = len(param_keylist)

        # for key_name in self.GS.cv_results_.keys():
        #     key_string=str(key_name)
        #     if key_string.find('param_')!=-1:
        #         if key_string.find('hidden')!=-1:
        #             self.grid_search_params.append(np.array(list(map(list,self.GS.cv_results_[key_name].data))))
        #             t1,t2=np.unique(self.grid_search_params[self.num_plot_axis],axis=0,return_inverse=True)
        #         else:
        #             self.grid_search_params.append(self.GS.cv_results_[key_name].data.astype('float'))
        #             t1, t2 = np.unique(self.grid_search_params[self.num_plot_axis], return_inverse=True)
        #         self.search_param_list.append(key_string)
        #         self.grid_search_params_unique.append(t1)
        #         self.grid_search_params_unique_indices.append(t2)
        #         self.num_plot_axis = self.num_plot_axis + 1

        if self.num_plot_axis == 1:
            xaxis = self.param_dict[param_keylist[0]]
            fig, ax = plt.subplots()
            ax.plot(xaxis, self.grid_search_result['mean_train_score'], label='Mean Train Score')
            ax.plot(xaxis, self.grid_search_result['mean_test_score'], label='Mean Test Score')
            ax.set_xlabel(param_keylist[0])
            ax.set_ylabel('Accuracy')
            ax.legend(loc='best')
            xtick_values = ['']
            for i in xaxis:
                xtick_values.append(str(i))
            ax.set_xticklabels(xtick_values)
            ax.set_title(self.estimator_name + ' ' + self.dataset_name)
            fig.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + 'Gridsearch_Train_Test.png'),
                        format='png')
        elif self.num_plot_axis == 2:
            xaxis = self.param_dict[param_keylist[0]]
            yaxis = self.param_dict[param_keylist[1]]
            rx = len(xaxis)
            ry = len(yaxis)

            xtick_values = []
            for i in xaxis:
                xtick_values.append(str(i))

            ytick_values = ['']
            for i in yaxis:
                ytick_values.append(str(i))

            fig, ax = plt.subplots(nrows=1, ncols=2)
            Train = np.reshape(self.grid_search_result['mean_train_score'], [rx, ry])
            Test = np.reshape(self.grid_search_result['mean_test_score'], [rx, ry])
            Train_min = np.round(np.min(Train) * 10.0) / 10.0
            Train_max = np.round(np.max(Train) * 10.0) / 10.0
            Test_min = np.round(np.min(Test) * 10.0) / 10.0
            Test_max = np.round(np.max(Test) * 10.0) / 10.0

            mt0 = ax[0].matshow(Train, vmin=0.35, vmax=1, cmap=plt.get_cmap('viridis'))
            ax[0].set_xlabel(param_keylist[0])
            ax[0].set_ylabel(param_keylist[1])
            mt1 = ax[1].matshow(Test, vmin=0.35, vmax=1, cmap=plt.get_cmap('viridis'))
            ax[1].set_xlabel(param_keylist[0])
            ax[1].set_ylabel(param_keylist[1])

            for axi in ax:
                axi.set_yticks(ytick_values)
                axi.xaxis.tick_bottom()
                axi.set_xticklabels(xtick_values)
            fig.colorbar(mt1, ax=axi, orientation='horizontal')
            axi.set_title(self.estimator_name + ' ' + self.dataset_name)
            fig.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + 'Gridsearch_Train_Test.png'),
                        format='png')
        elif self.num_plot_axis == 3:

            xaxis = self.param_dict[param_keylist[0]]
            yaxis = self.param_dict[param_keylist[1]]
            zaxis = self.param_dict[param_keylist[2]]
            rx = len(xaxis)
            ry = len(yaxis)
            rz = len(zaxis)
            Train = np.reshape(self.grid_search_result['mean_train_score'], [rx, ry, rz])
            Test = np.reshape(self.grid_search_result['mean_test_score'], [rx, ry, rz])
            Train_min = np.round(np.min(Train) * 10.0) / 10.0
            Train_max = np.round(np.max(Train) * 10.0) / 10.0
            Test_min = np.round(np.min(Test) * 10.0) / 10.0
            Test_max = np.round(np.max(Test) * 10.0) / 10.0
            Tmin = np.min([Test_min, Train_min])
            Tmax = np.min([Train_max, Test_max])

            xtick_values = []
            for i in xaxis:
                xtick_values.append(str(i))

            ytick_values = ['']
            for i in yaxis:
                ytick_values.append(str(i))
            ztick_values = ['']
            for i in zaxis:
                ztick_values.append(str(i))

            numrow = np.int(np.round(np.sqrt(rx)))
            numcol = np.int(np.ceil(rx / np.float(numrow)))
            if numrow == 1:
                ax = np.expand_dims(ax, axis=0)
            fig0, ax = plt.subplots(nrows=numrow, ncols=numcol, figsize=(12, 6))
            row = -1
            for nrow in np.arange(rx):
                col = np.mod(nrow, numcol)
                if col == 0:
                    row = row + 1
                mt = ax[row, col].matshow(np.squeeze(Train[nrow, :, :]), vmin=0.35, vmax=1,
                                          cmap=plt.get_cmap('viridis'))
                ax[row, col].set_yticklabels(ytick_values)
                ax[row, col].xaxis.tick_bottom()
                ax[row, col].set_xticklabels(ztick_values)
                ax[row, col].set_xlabel(param_keylist[2])
                if col == 0:
                    ax[row, col].set_ylabel(param_keylist[1])
                ax[row, col].set_title(xtick_values[nrow] + '_Train')
            fig0.colorbar(mt, ax=ax[row, col], orientation='horizontal')

            fig0.suptitle(self.estimator_name + ' ' + self.dataset_name)
            fig0.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + 'Gridsearch_Train.png'), format='png')
            fig1, ax1 = plt.subplots(nrows=numrow, ncols=numcol, figsize=(12, 6))
            if numrow == 1:
                ax1 = np.expand_dims(ax1, axis=0)
            row = -1
            for nrow in np.arange(rx):
                col = np.mod(nrow, numcol)
                if col == 0:
                    row = row + 1
                mt = ax1[row, col].matshow(np.squeeze(Test[nrow, :, :]), vmin=0.35, vmax=1,
                                           cmap=plt.get_cmap('viridis'))
                ax1[row, col].set_yticklabels(ytick_values)
                ax1[row, col].xaxis.tick_bottom()
                ax1[row, col].set_xticklabels(ztick_values)
                ax1[row, col].set_xlabel(param_keylist[2])
                if col == 0:
                    ax1[row, col].set_ylabel(param_keylist[1])
                ax1[row, col].set_title(xtick_values[nrow] + '_Test')
            fig1.colorbar(mt, ax=ax1[row, col], orientation='horizontal')

            fig1.suptitle(self.estimator_name + ' ' + self.dataset_name)

            fig1.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + 'Gridsearch_Test.png'), format='png')

    def table_results_cvgridsearch(self):
        self.grid_search_result = dict()
        self.grid_search_result['mean_test_score'] = self.GS.cv_results_['mean_test_score']
        self.grid_search_result['mean_train_score'] = self.GS.cv_results_['mean_train_score']
        self.grid_search_result['mean_fit_time'] = self.GS.cv_results_['mean_fit_time']
        df = pd.DataFrame()

        param_keylist = list(self.param_dict.keys())
        for key_name in self.GS.cv_results_.keys():
            key_string = str(key_name)
            if key_string.find('param_') != -1:
                df[key_string.strip('param')] = self.GS.cv_results_[key_name].data
        df['Train Score'] = self.GS.cv_results_['mean_train_score']
        df['Test Score'] = self.GS.cv_results_['mean_test_score']
        df['Fit Time'] = self.GS.cv_results_['mean_fit_time']
        df.to_html(self.estimator_name + '.html')
        if cmd_exists('wkhtmltoimage'):
            callstr = 'wkhtmltoimage -f png --width 0 ' + self.estimator_name + '.html ' + os.path.join(self.Fig_dir,
                                                                                                        self.estimator_state_string + 'Gridsearch.png')
            call(callstr, shell=True)

    def Dump_table(self, df, table_name=None):
        if table_name == None:
            table_name = self.estimator_state_string
        df.to_html(self.estimator_name + '.html')
        df.to_html(table_name + '.html')
        if cmd_exists('wkhtmltoimage'):
            callstr = 'wkhtmltoimage -f png --width 0 ' + self.estimator_name + '.html ' + os.path.join(self.Fig_dir,
                                                                                                        self.estimator_state_string + 'Gridsearch.png')
            call(callstr, shell=True)

    def plot_learning_curve(self, skip_train=1,label_string='nc'):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """

        self.axlc.set_title(self.estimator_state_string)
        # if self.ylim is not None:
        #    plt.ylim(*self.ylim)
        self.axlc.set_xlabel("Training examples")
        self.axlc.set_ylabel("Score")
        X_train = self.X_train[0:-1:skip_train, :]
        y_train = self.y_train[0:-1:skip_train]

        train_sizes, train_scores, test_scores = learning_curve(
            self.estimator, X_train, y_train, cv=self.cv, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        self.axlc.grid()

        self.axlc.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        )
        self.axlc.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
        labelT="Train " + label_string
        self.axlc.plot(train_sizes, train_scores_mean, 'o-',
                label=labelT)
        labelCV="CV "+label_string
        self.axlc.plot(train_sizes, test_scores_mean, 'o-',
                label=labelCV)

        self.axlc.legend(loc="best")
        self.figlc.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + '.png'), format='png')

    def create_estimator_state_string(self):
        self.estimator_state_string = self.estimator_name + self.estimator_param_string + '_FT_' + str(
            self.feature_extract) + self.dataset_name
        # for key in self.param_dict:
        #    self.estimator_state_string=self.estimator_state_string+ '_' + key +'_' + str(self.param_dict[key])

        sl_str = ''
        # for s in self.select_labels:
        #    sl_str = sl_str + str(s)
        # self.estimator_state_string = self.estimator_state_string + self.dataset_name + '_labels_' + sl_str

    def Datafit(self, skip_train=10):
        self.skip_train = skip_train
        if self.USL_SL:
            self.estimator.fit(self.X_train[::skip_train, :], self.y_train[::skip_train])
        else:
            self.estimator.fit(self.X_train[::skip_train, :])

    def Predict(self, data=None, label=None):
        if data.all() == None:
            data = self.X_test
            label = self.y_test
        if self.USL_SL:
            return self.estimator.predict(data, label)
        else:
            return self.estimator.predict(data)


class USL_estimator(Learner):
    def __init__(self, estimator_name, param_dict, Data_dir, Fig_dir, dataset_name, cv=None, USL_SL_Flag=False,
                 n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), feature_extract=True, select_labels=[0, 1],
                 normalize_data=True, DoGridSearch=False, load_data=True):
        super().__init__(estimator_name=estimator_name, param_dict=param_dict, Data_dir=Data_dir, Fig_dir=Fig_dir,
                         dataset_name=dataset_name, cv=cv, n_jobs=n_jobs, USL_SL_Flag=USL_SL_Flag,
                         train_sizes=train_sizes,
                         feature_extract=feature_extract, select_labels=select_labels, normalize_data=normalize_data,
                         DoGridSearch=DoGridSearch, load_data=load_data)
        if DoGridSearch:
            self.create_estimator_no_params()
        else:
            self.create_estimator()
        self.create_estimator_state_string()
        self.estimator_state_string_pipeline = self.estimator_state_string

    def create_estimator_param_string(self):
        curr_params = self.estimator.get_params()
        if hasattr(self.estimator, 'n_clusters'):
            n_components = curr_params['n_clusters']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
        elif hasattr(self.estimator, 'n_components'):
            n_components = curr_params['n_components']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'

    def create_estimator(self):
        if self.estimator_name == 'K-means':
            n_components = self.param_dict['n_components']
            ninit = self.param_dict['n_init']
            self.estimator = cluster.KMeans(algorithm='full', n_clusters=n_components, n_init=ninit, n_jobs=self.n_jobs)
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
        elif self.estimator_name == 'EM':
            n_components = self.param_dict['n_components']
            covariance_type = 'full'
            max_iter = self.param_dict['max_iter']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
            self.estimator = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                                     max_iter=max_iter)
        elif self.estimator_name == 'PCA':
            n_components = self.param_dict['n_components']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
            self.estimator = decomposition.PCA(n_components=n_components)
        elif self.estimator_name == 'ICA':
            n_components = self.param_dict['n_components']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
            self.estimator = decomposition.FastICA(n_components=n_components, max_iter=1000)
        elif self.estimator_name == 'Random_Projection':
            n_components = self.param_dict['n_components']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
            self.estimator = random_projection.GaussianRandomProjection(n_components=n_components)
        elif self.estimator_name == 'Dictionary_Learning':
            n_components = self.param_dict['n_components']
            self.estimator_param_string = '_n_components_' + str(n_components) + '_'
            alpha = self.param_dict['alpha']
            self.estimator = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha,batch_size=20)

    def create_estimator_no_params(self):
        if self.estimator_name == 'K-means':
            self.estimator = cluster.KMeans(n_jobs=self.n_jobs)
        elif self.estimator_name == 'EM':
            self.estimator = mixture.GaussianMixture()
        elif self.estimator_name == 'PCA':
            self.estimator = decomposition.PCA()
        elif self.estimator_name == 'ICA':
            self.estimator = decomposition.FastICA()
        elif self.estimator_name == 'Random_Projection':
            self.estimator = random_projection.gaussian_random_matrix()
        elif self.estimator_name == 'Dictionary_Learning':
            self.estimator = decomposition.DictionaryLearning()

    def label_purity(self, Assigned_label, true_label, figname=None, figlabel=None):

        # Assigned_label=self.estimator.predict(self.X_train)
        if figname == None:
            figname = self.estimator_state_string
        if figlabel == None:
            figlabel = self.estimator_state_string
        Assigned_label_unique = np.unique(Assigned_label)
        True_label_unique = np.unique(true_label)
        self.Label_Purity = np.zeros((Assigned_label_unique.size, True_label_unique.size))

        for i in Assigned_label_unique:
            index = np.where(Assigned_label == i)[0]
            True_label_in_group = self.y_train[index]
            True_label_population = np.unique(True_label_in_group)
            for j in True_label_population:
                index2 = np.where(True_label_in_group == j)[0]
                self.Label_Purity[i, j] = index2.size
        Label_list = []
        [Label_list.append('True_label ' + str(i)) for i in True_label_unique]
        self.Label_Purity_dataframe = pd.DataFrame(data=self.Label_Purity, columns=Label_list)
        self.Dump_table(df=self.Label_Purity_dataframe, table_name=figlabel)
        self.Label_Purity_Percentage = np.max(self.Label_Purity, axis=0) / np.sum(self.Label_Purity, axis=0)
        self.ax0.plot(self.Label_Purity_Percentage, label=figlabel)
        self.ax0.set_xlabel('Class Label')
        self.ax0.set_ylabel('Label Purity')
        self.ax0.set_title(self.dataset_name)
        self.ax0.legend(loc='best')
        self.fig0.savefig(os.path.join(self.Fig_dir, figname + '.png'), format='png')

    def pipeline_and_predict_label_purity(self, estimator_object, data=None, label=None):
        if data.all() == None:
            data = self.X_test
            label = self.y_test
        self.pipeline = make_pipeline(estimator_object.estimator, self.estimator)
        self.pipeline.fit(self.X_train[::self.skip_train, :])
        self.estimator_state_string_pipeline = self.estimator_state_string + estimator_object.estimator_state_string
        estimator_object_params = estimator_object.estimator.get_params()
        figlabel = estimator_object.estimator_name + '_n_vector_' + str(
            estimator_object_params['n_components']) + self.estimator_name
        self.label_purity(Assigned_label=self.pipeline.predict(data), true_label=label,
                          figname=self.estimator_state_string_pipeline, figlabel=figlabel)

    def display_components(self):
        if hasattr(self.estimator, 'components_'):
            components = self.estimator.components_
        elif hasattr(self.estimator, 'cluster_centers_'):
            components = self.estimator.cluster_centers_
        elif hasattr(self.estimator, 'means_'):
            components = self.estimator.means_
        else:
            return
        num_comp = components.shape[0]
        nx = int(np.round(np.sqrt(num_comp)))
        ny = int(np.ceil(num_comp / nx))
        fig1, ax1 = plt.subplots(nrows=ny, ncols=nx)
        imgsize = int(np.sqrt(components.shape[1]))
        counter = -1
        for i in np.arange(ny):
            for j in np.arange(nx):
                counter = counter + 1
                if (counter > components.shape[0] - 1):
                    break
                ax1[i, j].imshow(components[counter, :].reshape(imgsize, imgsize))
                ax1[i, j].axis('off')

        fig1.savefig(os.path.join(self.Fig_dir, self.estimator_state_string + '_components.png'), format='png')

    def transform_error(self,data=None,label=None):

        data_inv_transform=self.estimator.inverse_transform(self.estimator.transform(data))
        #else:
        # transformed_data=self.estimator.transform(data)
        # transformation_matrix=self.estimator.components_
        # transformation_matrix_tpose=np.transpose(transformation_matrix)
        # transformation_matrix_pseudoinv=np.linalg.pinv(np.matmul(transformation_matrix_tpose,transformation_matrix))
        # data_inv_transform=np.matmul(np.matmul(transformed_data,transformation_matrix),transformation_matrix_pseudoinv)
        error=np.mean(np.abs(data-data_inv_transform)**2,axis=1)
        label_unique=np.unique(label)
        error_index=np.zeros(label_unique.shape)
        figlabel= self.estimator_name + self.estimator_param_string
        for i in label_unique:
            index = np.where(label == i)[0]
            error_index[i]=np.mean(error[index])
        self.ax1.plot(error_index, label=figlabel)
        self.ax1.set_xlabel('Class Label')
        self.ax1.set_ylabel('Class transform error')
        self.ax1.set_title(self.dataset_name)
        self.ax1.legend(loc='best')
        figname = self.estimator_state_string
        self.fig1.savefig(os.path.join(self.Fig_dir, figname + '.png'), format='png')


    def predict_label_purity(self, data=None, label=None):
        figlabel = self.estimator_name + self.estimator_param_string
        self.label_purity(Assigned_label=self.Predict(data=data), true_label=label, figlabel=figlabel)


class SL_estimator(Learner):
    def __init__(self, estimator_name, param_dict, Data_dir, Fig_dir, dataset_name, cv=None, USL_SL_Flag=False,
                 n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), feature_extract=True, select_labels=[0, 1],
                 normalize_data=True, DoGridSearch=False):
        super().__init__(estimator_name=estimator_name, param_dict=param_dict, Data_dir=Data_dir, Fig_dir=Fig_dir,
                         dataset_name=dataset_name, cv=cv, n_jobs=n_jobs, USL_SL_Flag=USL_SL_Flag,
                         train_sizes=train_sizes,
                         feature_extract=feature_extract, select_labels=select_labels, normalize_data=normalize_data,
                         DoGridSearch=DoGridSearch)
        if self.DoGridSearch:
            self.create_estimator_no_params()
        else:
            self.create_estimator()
        self.create_estimator_state_string()

    def create_estimator(self):
        if self.estimator_name == 'Decision_Tree':
            max_depth = self.param_dict['max_depth']
            criterion = self.param_dict['criterion']
            min_samples_split = self.param_dict['min_samples_split']
            min_samples_leaf = self.param_dict['min_samples_leaf']
            self.estimator = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                         min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf)
        elif self.estimator_name == 'KNN':
            n_neighbors = self.param_dict['n_neighbors']
            self.estimator = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        elif self.estimator_name == 'MLP':
            hidden_layer_sizes = param_dict['hidden_layer_sizes']
            learning_rate_init = param_dict['learning_rate_init']
            learning_rate = param_dict['learning_rate']
            self.estimator = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                          learning_rate=learning_rate,
                                                          learning_rate_init=learning_rate_init)
        elif self.estimator_name == 'Random Forest':
            max_depth = self.param_dict['max_depth']
            criterion = self.param_dict['criterion']
            self.estimator = RFC(max_depth=max_depth, criterion=criterion)
        elif self.estimator_name == 'Adaboost':
            n_estimator = self.param_dict['n_estimator']
            learning_rate = self.param_dict['learning_rate']
            self.estimator = AdaB(
                base_estimator=tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, min_samples_split=80),
                n_estimators=n_estimator, learning_rate=learning_rate)
        elif self.estimator_name == 'SVM':
            gamma = self.param_dict['gamma']
            Cval = self.param_dict['C']
            self.estimator = SVC(C=Cval, gamma=gamma)
        elif self.estimator_name == 'SVMP':
            gamma = self.param_dict['gamma']
            Cval = self.param_dict['C']
            kernel = self.param_dict['kernel']
            degree = self.param_dict['degree']
            self.estimator = SVC(C=Cval, gamma=gamma, kernel=kernel, degree=degree)

        self.estimator_param_string = [self.estimator_param_string + key + '_' + str(param_dict[key]) for key in
                                       param_dict]
        self.estimator_param_string = '_'.join(self.estimator_param_string)

    def create_estimator_no_params(self):
        if self.estimator_name == 'Decision_Tree':
            self.estimator = tree.DecisionTreeClassifier(criterion='entropy')
        elif self.estimator_name == 'KNN':
            self.estimator = neighbors.KNeighborsClassifier()
        elif self.estimator_name == 'MLP':
            self.estimator = neural_network.MLPClassifier(learning_rate='invscaling')
        elif self.estimator_name == 'Random Forest':
            self.estimator = RFC(criterion='entropy')
        elif self.estimator_name == 'Adaboost':
            self.estimator = AdaB(
                base_estimator=tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, min_samples_split=80))
        elif self.estimator_name == 'SVM':
            self.estimator = SVC()
        elif self.estimator_name == 'SVMP':
            self.estimator = SVC()

    def transform_data(self, estimator_object):
        # self.pipeline = make_pipeline(estimator_object.estimator, self.estimator)
        estimator_object.estimator.fit(self.X_train)
        self.estimator_state_string = self.estimator_state_string + estimator_object.estimator_state_string
        self.X_train_untransformed = self.X_train
        self.X_test_untransformed = self.X_test
        if estimator_object.estimator_name=='EM':
            self.X_train=estimator_object.estimator.predict_proba(self.X_train)
            self.X_test = estimator_object.estimator.predict_proba(self.X_test)
        else:
            self.X_train = estimator_object.estimator.transform(self.X_train)
            self.X_test = estimator_object.estimator.transform(self.X_test)


def Get_Param_Dict(estimator_name, DoGridSearch):
    if DoGridSearch:
        param_dict = OD()
        if estimator_name == 'Decision_Tree':
            param_dict['max_depth'] = [5, 7, 9, 11]
            param_dict['min_samples_split'] = [20, 40, 80]
            param_dict['min_samples_leaf'] = [10, 20, 40]
        elif estimator_name == 'KNN':
            param_dict['n_neighbors'] = [5, 9, 13, 15]
            param_dict['weights'] = ['uniform', 'distance']

        elif estimator_name == 'MLP':
            param_dict['hidden_layer_sizes'] = [(20, 10), (40, 20), (80, 40), (40, 20, 10)]
            param_dict['learning_rate_init'] = [0.0001, 0.001, 0.01]
            param_dict['max_iter'] = [5000, 10000]
        elif estimator_name == 'Random Forest':
            param_dict['max_depth'] = [5, 7, 9, 11, 13]
            param_dict['min_samples_split'] = [20, 40, 80]
            param_dict['min_samples_leaf'] = [10, 20, 40]
        elif estimator_name == 'Adaboost':
            param_dict['n_estimators'] = [25, 100, 200]
            param_dict['learning_rate'] = [0.05, 0.25, 0.5, 1]

        elif estimator_name == 'SVM':
            param_dict['C'] = np.logspace(-2, 2, 5)
            param_dict['gamma'] = [0.5, 1, 5, 10]
        elif estimator_name == 'SVMP':
            param_dict['kernel'] = ['poly', 'rbf']
            param_dict['C'] = np.logspace(-2, 2, 3)
            param_dict['gamma'] = [0.5, 1, 5, 10]
            param_dict['degree'] = [1, 2, 3]
        elif estimator_name == 'K-Means':
            param_dict['n_components'] = [7, 8, 9, 10, 11]
            param_dict['n_init'] = 10
        elif estimator_name == 'EM':
            param_dict['n_components'] = [7, 8, 9, 10, 11]
            param_dict['max_iter'] = 100
        elif estimator_name == 'PCA':
            param_dict['n_components'] = [100, 200, 300, 400, 500]
    else:
        param_dict = OD()
        if estimator_name == 'Decision_Tree':
            param_dict = {'max_depth': 9, 'criterion': "entropy", 'min_samples_split': 40, 'min_samples_leaf': 20}
        elif estimator_name == 'KNN':
            param_dict = {'n_neighbors': 9}
        elif estimator_name == 'MLP':
            param_dict = {'hidden_layer_sizes': (20, 20), 'learning_rate': 'invscaling',
                          'learning_rate_init': 0.0002,
                          'max_iter': 10000}
        elif estimator_name == 'Adaboost':
            param_dict = {'n_estimators': 100, 'learning_rate': 0.25}
        elif estimator_name == 'SVM':
            param_dict = {'C': 10, 'gamma': 1}
        elif estimator_name == 'SVM':
            param_dict = {'C': 10, 'gamma': 1}
        elif estimator_name == 'K-means':
            param_dict['n_components'] = 10
            param_dict['n_init'] = 10
        elif estimator_name == 'EM':
            param_dict['n_components'] = 10
            param_dict['max_iter'] = 100
        elif estimator_name == 'PCA':
            param_dict['n_components'] = 100
        elif estimator_name == 'ICA':
            param_dict['n_components'] = 100
        elif estimator_name == 'Random_Projection':
            param_dict['n_components'] = 50
        elif estimator_name == 'Dictionary_Learning':
            param_dict['n_components'] = 50
            param_dict['alpha'] = 0.8

    return param_dict


if __name__ == "__main__":
    Data_Dir = 'Dataset'
    Fig_Dir = 'Figures'
    select_labels=[0,1,2,3]
    SL_estimator_name_set = ['MLP']
    USL_estimator_name_set = ['K-means', 'EM', 'PCA', 'ICA', 'Random_Projection', 'Dictionary_Learning']
    process_flag = 5  # 1=clustering 2=dim red 3=dim red+cluster 4=dim red+ nn 5=cluster + nn
    if process_flag == 1:
        estimator_name_set = ['K-means', 'EM']
    elif process_flag == 2:
        estimator_name_set = ['PCA','ICA''Random_Projection', 'Dictionary_Learning']
    elif process_flag == 3:
        estimator_name_set = ['K-means', 'EM']
        pipeline_estimator_name_set=['PCA','ICA''Random_Projection', 'Dictionary_Learning']
    elif process_flag>=4:
        estimator_name_set=SL_estimator_name_set
    if set(estimator_name_set) <= set(USL_estimator_name_set):
        USL_SL_Flag = False
    else:
        USL_SL_Flag = True
    for estimator_name in estimator_name_set:
        dataset_name = 'Mnist'
        if dataset_name == 'Cifar-10':
            if os.path.isdir(os.path.join(os.getcwd(), Data_Dir, 'cifar-10-batches-py')) is False:
                download_cifar()
        elif dataset_name == 'Mnist':
            if os.path.isfile(os.path.join(os.getcwd(), Data_Dir, 'mnist_train.csv')) is False:
                download_mnist()

        DoGridSearch = False

        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
        param_dict = Get_Param_Dict(estimator_name=estimator_name, DoGridSearch=DoGridSearch)


        if process_flag<=3:
            USL_estimator_object = USL_estimator(param_dict=param_dict, estimator_name=estimator_name,
                                                 USL_SL_Flag=USL_SL_Flag, Data_dir=Data_Dir, feature_extract=False,
                                                 select_labels=select_labels,
                                                 Fig_dir=Fig_Dir, dataset_name=dataset_name, cv=cv, n_jobs=8,
                                                 normalize_data=False, DoGridSearch=DoGridSearch)
        if process_flag > 2:
            pipeline_estimator_name = 'EM'
            param_dict_pipeline = Get_Param_Dict(estimator_name=pipeline_estimator_name, DoGridSearch=DoGridSearch)
            USL_estimator_object_pipeline = USL_estimator(param_dict=param_dict_pipeline,
                                                          estimator_name=pipeline_estimator_name,
                                                          USL_SL_Flag=USL_SL_Flag, Data_dir=Data_Dir,
                                                          feature_extract=False,
                                                          select_labels=select_labels,
                                                          Fig_dir=Fig_Dir, dataset_name=dataset_name, cv=cv, n_jobs=8,
                                                          normalize_data=False, DoGridSearch=DoGridSearch,
                                                          load_data=False)

        if USL_SL_Flag:
            SL_estimator_object = SL_estimator(param_dict=param_dict, estimator_name=estimator_name,
                                               USL_SL_Flag=USL_SL_Flag, Data_dir=Data_Dir, feature_extract=False,
                                               select_labels=select_labels,
                                               Fig_dir=Fig_Dir, dataset_name=dataset_name, cv=cv, n_jobs=8,
                                               normalize_data=False, DoGridSearch=DoGridSearch)
        if DoGridSearch:
            start_time = timeit.default_timer()
            USL_estimator_object.cvgridsearch(skip_train=2)
            USL_estimator_object.elapsed_time = timeit.default_timer() - start_time
            print("Elapsed Time=", USL_estimator_object.elapsed_time)
        else:
            start_time = timeit.default_timer()
            if USL_SL_Flag:
                if process_flag==4:
                    n_components1 = [10,50,100]
                elif process_flag==5:
                    n_components1 = [8, 10, 12]
                for i in np.arange(len(n_components1)):
                    print(i)
                    curr_n_components = n_components1[i]
                    if hasattr(USL_estimator_object_pipeline.estimator, 'n_components'):
                        USL_estimator_object_pipeline.estimator.set_params(n_components=curr_n_components)
                    elif hasattr(USL_estimator_object_pipeline.estimator, 'n_clusters'):
                        USL_estimator_object_pipeline.estimator.set_params(n_clusters=curr_n_components)
                    USL_estimator_object_pipeline.create_estimator_param_string()
                    USL_estimator_object_pipeline.create_estimator_state_string()
                    SL_estimator_object.create_estimator_state_string()
                    SL_estimator_object.transform_data(estimator_object=USL_estimator_object_pipeline)
                    SL_estimator_object.plot_learning_curve(skip_train=2,label_string='num_comp_'+str(curr_n_components))
                    SL_estimator_object.X_train=SL_estimator_object.X_train_untransformed
                    SL_estimator_object.X_test = SL_estimator_object.X_test_untransformed
                    #USL_estimator_object_pipeline.display_components()
            else:
                if process_flag == 1:
                    if dataset_name=='Mnist':
                        n_components1 = [6, 8, 10, 12, 14]
                    else:
                        n_components1 = [3, 4, 5]
                elif process_flag == 2:
                    n_components1 = [6, 10, 50, 100]
                    for i in np.arange(len(n_components1)):
                        print(i)
                        curr_n_components = n_components1[i]
                        if hasattr(USL_estimator_object.estimator, 'n_components'):
                            USL_estimator_object.estimator.set_params(n_components=curr_n_components)
                        elif hasattr(USL_estimator_object.estimator, 'n_clusters'):
                            USL_estimator_object.estimator.set_params(n_clusters=curr_n_components)
                        USL_estimator_object.create_estimator_param_string()
                        USL_estimator_object.create_estimator_state_string()
                        USL_estimator_object.Datafit()
                        USL_estimator_object.display_components()
                        if process_flag == 1:
                            USL_estimator_object.predict_label_purity(data=USL_estimator_object.X_train,
                                                                      label=USL_estimator_object.y_train)
                        elif process_flag==2:
                            transform_attr = hasattr(USL_estimator_object.estimator, 'inverse_transform')
                            if transform_attr:
                                USL_estimator_object.transform_error(data=USL_estimator_object.X_train,label=USL_estimator_object.y_train)
                elif process_flag == 3:
                    n_components = [6, 10, 50,100]
                    for i in np.arange(len(n_components)):
                        print(i)
                        curr_n_components = n_components[i]
                        USL_estimator_object_pipeline.estimator.set_params(n_components=curr_n_components)
                        USL_estimator_object_pipeline.create_estimator_param_string()
                        USL_estimator_object_pipeline.create_estimator_state_string()
                        USL_estimator_object.pipeline_and_predict_label_purity(
                            estimator_object=USL_estimator_object_pipeline,
                            data=USL_estimator_object.X_train,
                            label=USL_estimator_object.y_train)



                # class_name = []
            # if dataset_name == 'Mnist':
            #     for i in np.arange(10):
            #         class_name.append(str(i))
            # elif dataset_name=='Cifar-10':
            #     for i in np.arange(4):
            #         class_name.append(str(i))

            # dot_data = tree.export_graphviz(SL_estimator_object.estimator,out_file=None, class_names=class_name,
            #                 filled=True, rounded=True,
            #                 special_characters=True)
            # dot_data=tree.export_graphviz(SL_estimator_object.estimator,out_file=None)
            # graph = graphviz.Source(dot_data)
            # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

            # graph.render('Dataset_'+dataset_name+'_'+SL_estimator_object.estimator_state_string)
            # DTS.DT_Structure(SL_estimator_object.estimator, X_test=np.reshape(SL_estimator_object.X_test[0, :], (1, -1)))
            # USL_estimator_object.plot_learning_curve(skip_train=2)
            if process_flag<=3:
                USL_estimator_object.elapsed_time = timeit.default_timer() - start_time
                print("Elapsed Time=", USL_estimator_object.elapsed_time)
                del USL_estimator_object
            else:
                USL_estimator_object_pipeline.elapsed_time = timeit.default_timer() - start_time
                print("Elapsed Time=", USL_estimator_object_pipeline.elapsed_time)



    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.

# clf=tree.DecisionTreeClassifier()
# clf=clf.fit(train_data,train_target)
# ypred=clf.predict(test_data)
# accuracy=skl.metrics.accuracy_score(test_target,ypred)
# print("Accuracy= ",accuracy)

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("abalone")
