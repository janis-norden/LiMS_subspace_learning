import numpy as np
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
from plot_styles import load_general_styles, load_gravitational_waves_styles, load_prednisone_3D_styles, load_restricted_unrestricted_styles
import src.utility as utility
from scipy.stats import multivariate_normal
from src.data_box import DataBox
from copy import deepcopy


# Data generation pipeline -----------------------

def plot_data_generation_pipeline(label_fontsize=14, title_fontsize=16, legend_fontsize=12, marker_size=10, scatter_area=10., line_width=5, tick_fontsize=10):
    '''
    Data generation pipeline with 2D plots
    '''

    # Load plotting styles
    general_styles = load_general_styles()
    pred_styles = load_prednisone_3D_styles()

    # Load data box from file
    data_box_full = DataBox().load_data('/interim/prednisone_3D_full_finite_mixture.pckl')

    # Create the plot
    column_width = general_styles['column_width']
    fig = plt.figure(figsize=(2 * column_width, (2 * column_width) / 4))

    # 1) Spirals in parameter space
    labels = data_box_full.get_labels()
    idx_C0 = [idx for idx, value in enumerate(labels) if value == 0]
    idx_C1 = [idx for idx, value in enumerate(labels) if value == 1]
    parameters = data_box_full.get_parameters()
    params_C0 = parameters[idx_C0, :]
    params_C1 = parameters[idx_C1, :]

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(params_C0[:, 0], -params_C0[:, 1] + params_C0[:, 2],
                c=pred_styles['class_colours'][0], marker=pred_styles['class_markers'][0],
                s=scatter_area, label=pred_styles['class_labels'][0])
    ax1.scatter(params_C1[:, 0], -params_C1[:, 1] + params_C1[:, 2],
                c=pred_styles['class_colours'][1], marker=pred_styles['class_markers'][1],
                s=scatter_area, label=pred_styles['class_labels'][1])
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_xlabel('$k_{ex}$', fontsize=label_fontsize)
    ax1.set_ylabel('$k_{LP} - k_{PL}$', fontsize=label_fontsize)
    ax1.set_title('a) Ground truth parameters', fontsize=title_fontsize)
    ax1.set_xticks([0, 0.1, 0.2])
    ax1.set_yticks([-0.1, 0., 0.1])
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.legend(fontsize=legend_fontsize)

    # 2) Time series data
    ax2 = fig.add_subplot(1, 4, 2)
    timeseries = data_box_full.get_timeseries()
    for i in range(len(idx_C0)):
        label = pred_styles['class_labels'][0] if i == 0 else None
        ax2.plot(timeseries[idx_C0[i]][0, :], timeseries[idx_C0[i]][1, :],
                 color=pred_styles['class_colours'][0], marker=pred_styles['class_markers'][0],
                 markersize=marker_size, linewidth=line_width, alpha=0.3, label=label)
    for i in range(len(idx_C1)):
        label = pred_styles['class_labels'][1] if i == 0 else None
        ax2.plot(timeseries[idx_C1[i]][0, :], timeseries[idx_C1[i]][1, :],
                 color=pred_styles['class_colours'][1], marker=pred_styles['class_markers'][1],
                 markersize=marker_size, linewidth=line_width, alpha=0.3, label=label)
    ax2.set_xlabel('$t$', fontsize=label_fontsize)
    ax2.set_ylabel('$P(t)$', fontsize=label_fontsize)
    ax2.set_xlim(0, 240)
    ax2.set_ylim(0, 180)
    ax2.set_title('b) Time series', fontsize=title_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)

    # 3) Samples
    downsample_fact = 50
    class_samples = data_box_full._merge_samples_by_class()
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(class_samples[0][0::downsample_fact, 1], class_samples[0][0::downsample_fact, 2],
                c=pred_styles['class_colours'][0], marker=pred_styles['class_markers'][0],
                s=scatter_area, label=pred_styles['class_labels'][0])
    ax3.scatter(class_samples[1][0::downsample_fact, 1], class_samples[1][0::downsample_fact, 2],
                c=pred_styles['class_colours'][1], marker=pred_styles['class_markers'][1],
                s=scatter_area, label=pred_styles['class_labels'][1])
    ax3.set_xlim([0, 0.2])
    ax3.set_ylim([0, 0.2])
    ax3.set_xlabel('$k_{PL}$', fontsize=label_fontsize)
    ax3.set_ylabel('$k_{LP}$', fontsize=label_fontsize)
    ax3.set_title('c) Posterior samples', fontsize=title_fontsize)
    ax3.set_xticks([0, 0.1, 0.2])
    ax3.set_yticks([0, 0.1, 0.2])
    ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax3.legend(fontsize=legend_fontsize)

    # 4) Density estimates
    ax4 = fig.add_subplot(1, 4, 4)
    data_box_projected = data_box_full.project_density_estimates(np.array([[0,  0], [1,  0], [0,  1]]))
    density_estimates = data_box_projected.get_density_estimates()
    num_examples = len(labels)
    X, Y = np.meshgrid(np.linspace(0, 0.2, 100), np.linspace(0, 0.2, 100))
    pos = np.dstack((X, Y))

    Z_C0 = np.zeros(X.shape)
    Z_C1 = np.zeros(X.shape)
    Z_C2 = np.zeros(X.shape)

    for n in range(num_examples):
        num_comps = len(density_estimates[n]['mix_weights'])
        Z = np.zeros(X.shape)
        for k in range(num_comps):
            mu = density_estimates[n]['mu_array'][:, k]
            Sigma = density_estimates[n]['Sigma_array'][k, :, :]
            weight = density_estimates[n]['mix_weights'][k]
            Z += weight * multivariate_normal.pdf(pos, mu, Sigma)

        if labels[n] == 0:
            Z_C0 += Z
        elif labels[n] == 1:
            Z_C1 += Z
        else:
            Z_C2 += Z

        vmax = np.max(Z)
        levels = np.linspace(0.1 * vmax, vmax, num=10)

        linestyle = 'solid' if int(labels[n]) == 0 else 'dashed'
        ax4.contour(X, Y, Z, levels=levels, colors=pred_styles['class_colours'][int(labels[n])],
                    linestyles=linestyle, linewidths=line_width, alpha=0.2)

    ax4.plot(np.linspace(0.3, 0.4, 10), color=pred_styles['class_colours'][0],
             label=pred_styles['class_labels'][0], linestyle='-', linewidth=line_width, alpha=0.2)
    ax4.plot(np.linspace(0.3, 0.4, 10), color=pred_styles['class_colours'][1],
             label=pred_styles['class_labels'][1], linestyle='--', linewidth=line_width, alpha=0.2)

    ax4.set_xlim([0, 0.2])
    ax4.set_ylim([0, 0.2])
    ax4.set_xlabel('$k_{PL}$', fontsize=label_fontsize)
    ax4.set_ylabel('$k_{LP}$', fontsize=label_fontsize)
    ax4.set_title('d) Density estimates', fontsize=title_fontsize)
    ax4.set_xticks([0, 0.1, 0.2])
    ax4.set_yticks([0, 0.1, 0.2])
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax4.legend(loc='lower right', fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()

    return fig

# Experiment 1 -----------------------------------

def plot_accuracy_time_exp1_restricted_unrestricted(
    filename_restricted, 
    filename_unrestricted, 
    winner_takes_all=True, 
    avg_type='macro',
    axis_label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    boxplot_linewidth=1.5,
    text_fontsize=10
):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename_restricted + '.pckl', 'rb')
    exp1_data_restricted = pickle.load(f)
    f.close()

    f = open('../data/processed/' + filename_unrestricted + '.pckl', 'rb')
    exp1_data_unrestricted = pickle.load(f)
    f.close()

    # load plot styles and extract
    general_styles = load_general_styles()
    restricted_unrestricted_styles = load_restricted_unrestricted_styles()

    # change to format easily passable to seaborn
    df = create_df_exp1_restricted_unrestricted(exp1_data_restricted, exp1_data_unrestricted, winner_takes_all)

    # set title string
    title_string = 'Prednisone Spiral (winner_takes_all: ' + str(winner_takes_all) + ' | ' + avg_type + ' averages)'

    # remove runs with bad training behaviour
    df = df[(df.accuracy_train_macro > 0.9) | (df.dimension == 1)]

    # set time of dimension 3 to None
    df.loc[df['dimension'] == 3, 'time'] = None

    # Plot
    # create grouped boxplot
    column_width = general_styles['column_width']
    fig, ax1 = plt.subplots(figsize=(2 * column_width, (2 * column_width) / 4))
    
    # Boxplot for accuracy
    box_acc = sns.boxplot(
        x=df['dimension'], 
        y=df['accuracy_test_' + avg_type],
        hue=df['cost_func'],
        hue_order=restricted_unrestricted_styles['hue_order'],
        palette=restricted_unrestricted_styles['palette'],
        legend=True,
        showfliers=False,
        ax=ax1,
        linewidth=boxplot_linewidth
    )
    ax1.set(
        xlabel='subspace dimension $d^{\prime}$', 
        ylabel='test accuracy',
        ylim=[0, 1.01]
    )
    ax1.xaxis.label.set_size(axis_label_fontsize)
    ax1.yaxis.label.set_size(axis_label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Boxplot for time
    ax2 = ax1.twinx()
    box_time = sns.boxplot(
        x=df['dimension'], 
        y=df['time'],
        hue=df['cost_func'],
        hue_order=restricted_unrestricted_styles['hue_order'],
        palette=restricted_unrestricted_styles['palette'],
        legend=True,
        showfliers=False,
        fill=False,
        ax=ax2,
        linewidth=boxplot_linewidth
    )
    ax2.set(
        ylabel='time (s)',
        ylim=[0, 4000]
    )
    ax2.yaxis.label.set_size(axis_label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Adjust legend entries
    handles1, previous_labels = ax1.get_legend_handles_labels()
    ax1.legend(loc=(0.776, 0.45), handles=handles1, labels=['a) acc. - spherical $K_{max}=5$', 'b) acc. - spherical $K_{max}=20$', 'c) acc. - full $K_{max}=5$', 'd) acc. - full $K_{max}=20$'], fontsize=legend_fontsize)

    handles2, previous_labels = ax2.get_legend_handles_labels()
    ax2.legend(loc='lower right', handles=handles2, labels=['e) time - spherical $K_{max}=5$', 'f) time - spherical $K_{max}=20$', 'g) time - full $K_{max}=5$', 'h) time - full $K_{max}=20$'], fontsize=legend_fontsize)

    # Add text to image
    shift_small = 0.12
    shift_large = 1

    ax1.text(-0.2 - shift_small, 0.7, 'a)', fontsize=text_fontsize)
    ax1.text(0. - shift_small, 0.7, 'b)', fontsize=text_fontsize)
    ax1.text(0.2 - shift_small, 0.7, 'c)', fontsize=text_fontsize)
    ax1.text(0.4 - shift_small, 0.7, 'd)', fontsize=text_fontsize)
    ax1.text(-0.2 - shift_small, 0.05, 'e)', fontsize=text_fontsize)
    ax1.text(0. - shift_small, 0.05, 'f)', fontsize=text_fontsize)
    ax1.text(0.2 - shift_small, 0.015, 'g)', fontsize=text_fontsize)
    ax1.text(0.4 - shift_small, 0.015, 'h)', fontsize=text_fontsize)

    ax1.text(-0.2 - shift_small + shift_large, 0.94, 'a)', fontsize=text_fontsize)
    ax1.text(0. - shift_small + shift_large, 0.94, 'b)', fontsize=text_fontsize)
    ax1.text(0.2 - shift_small + shift_large, 0.94, 'c)', fontsize=text_fontsize)
    ax1.text(0.4 - shift_small + shift_large, 0.94, 'd)', fontsize=text_fontsize)
    ax1.text(-0.2 - shift_small + shift_large, 0.05, 'e)', fontsize=text_fontsize)
    ax1.text(0. - shift_small + shift_large, 0.05, 'f)', fontsize=text_fontsize)
    ax1.text(0.2 - shift_small + shift_large, 0.22, 'g)', fontsize=text_fontsize)
    ax1.text(0.4 - shift_small + shift_large, 0.22, 'h)', fontsize=text_fontsize)

    ax1.text(-0.2 - shift_small + 2 * shift_large, 0.94, 'a)', fontsize=text_fontsize)
    ax1.text(0. - shift_small + 2 * shift_large, 0.94, 'b)', fontsize=text_fontsize)
    ax1.text(0.2 - shift_small + 2 * shift_large, 0.94, 'c)', fontsize=text_fontsize)
    ax1.text(0.4 - shift_small + 2 * shift_large, 0.94, 'd)', fontsize=text_fontsize)

    #fig.suptitle(title_string)
    fig.tight_layout()
    plt.show()

    # print stats. on run time
    df_sphe_dim1 = df[(df['cost_func'].str.contains('spherical')) & (df['dimension']==1)]['time']
    df_sphe_dim2 = df[(df['cost_func'].str.contains('spherical')) & (df['dimension']==2)]['time']
    df_full_dim1 = df[(df['cost_func'].str.contains('full')) & (df['dimension']==1)]['time']
    df_full_dim2 = df[(df['cost_func'].str.contains('full')) & (df['dimension']==2)]['time']
    print('Spherical, dim=1: \n', df_sphe_dim1.describe(), '\n', '------------------------------------')
    print('Spherical, dim=2: \n', df_sphe_dim2.describe(), '\n', '------------------------------------')
    print('Full, dim=1: \n', df_full_dim1.describe(), '\n', '------------------------------------')
    print('Full, dim=2: \n', df_full_dim2.describe(), '\n', '------------------------------------')

    return fig

def create_df_exp1_restricted_unrestricted(exp1_data_restricted, exp1_data_unrestricted, winner_takes_all = True):
    '''unrolls information stored in exp1_data into a dataframe-like format that is easily readable by Seaborn'''

    # make dict in dataframe format
    df = {}
    df['dimension'] = []
    df['cost_func'] = []
    df['init_type'] = []

    df['time'] = []
    df['accuracy_train_micro'] = []
    df['accuracy_train_macro'] = []
    df['accuracy_test_micro'] = []
    df['accuracy_test_macro'] = []
    df['Grassmann_dist'] = []
    df['log_likelihood_train'] = []
    df['log_likelihood_test'] = []
    df['BIC'] = []
    df['AIC'] = []

    # unpack from exp1_data
    train_index = exp1_data_restricted['train_index']
    test_index = exp1_data_restricted['test_index']
    
    data_box_train = exp1_data_restricted['data_box_spherical'].select_examples(train_index)
    data_box_test = exp1_data_restricted['data_box_spherical'].select_examples(test_index)

    labels_train = data_box_train.get_labels()
    labels_test = data_box_test.get_labels()

    num_dim_ambient = exp1_data_restricted['data_box_spherical'].get_dynamical_system().num_params

    true_subspace = exp1_data_restricted['data_box_spherical'].get_subspace()
    V_true = np.vstack((true_subspace['v1'], true_subspace['v2'])).T

    # extract experimental info
    cols_vec = exp1_data_restricted['exp1_opts']['cols_vec']    
    num_rand_init = exp1_data_restricted['exp1_opts']['num_rand_init']

    # loop over data sets
    for data_type in ['restricted', 'unrestricted']:

        if data_type == 'restricted':
            exp1_data = exp1_data_restricted
        else:
            exp1_data = exp1_data_unrestricted

        # loop over dimensionalities
        for cols_idx in range(len(cols_vec)):

            # loop over cost function types
            for cost_func in ['spherical', 'full']:

                # loop over random inits
                for trial in range(num_rand_init):
                    
                    # record independent variables
                    df['dimension'].append(cols_vec[cols_idx])
                    df['cost_func'].append(cost_func + '_' + data_type)
                    df['init_type'].append('random')
                    
                    # record dependent variables
                    df['time'].append(exp1_data['outcomes'][cols_idx][cost_func][trial]['ssl_time'])

                    # extract predictions and evaluate accuracy
                    pht_predictions_train = exp1_data['outcomes'][cols_idx][cost_func][trial]['pht_predictions_train']
                    pht_predictions_test = exp1_data['outcomes'][cols_idx][cost_func][trial]['pht_predictions_test']

                    # calculate accuracies and append to df
                    df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)

                    # find and append Grassmann distance to true subspace (only for dim = 2)
                    if cols_idx == 1:
                        V_opt = exp1_data['outcomes'][cols_idx][cost_func][trial]['subspace_learned']['V_opt']
                        df['Grassmann_dist'].append(utility.GrassmannDist(V_true, V_opt))
                    else:
                        df['Grassmann_dist'].append(None)

                # add GMLVQ init
                df['dimension'].append(cols_vec[cols_idx])
                df['cost_func'].append(cost_func + '_' + data_type)
                df['init_type'].append('GMLVQ')
                
                df['time'].append(exp1_data['outcomes'][cols_idx][cost_func][num_rand_init]['ssl_time'])
                
                # extract predictions and evaluate accuracy
                pht_predictions_train = exp1_data['outcomes'][cols_idx][cost_func][num_rand_init]['pht_predictions_train']
                pht_predictions_test = exp1_data['outcomes'][cols_idx][cost_func][num_rand_init]['pht_predictions_test']
                
                # calculate accuracies and append to df
                df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)

                # find and append Grassmann distance to true subspace (only for dim = 2)
                if cols_idx == 1:
                    V_opt = exp1_data['outcomes'][cols_idx][cost_func][trial]['subspace_learned']['V_opt']
                    df['Grassmann_dist'].append(utility.GrassmannDist(V_true, V_opt))
                else:
                    df['Grassmann_dist'].append(None)

        # add results for ambient dimension
        for cost_func in ['spherical', 'full']:
            for init_type in ['random', 'GMLVQ']:

                df['dimension'].append(num_dim_ambient)
                df['cost_func'].append(cost_func + '_' + data_type)
                df['init_type'].append(init_type)
                
                df['time'].append(exp1_data['outcomes'][len(cols_vec)][cost_func][0]['ssl_time'])
                #df['accuracy'].append(exp4_data['outcomes_folds'][fold_idx][len(cols_vec)][cost_func][0]['accuracy'])

                # extract predictions and evaluate accuracy
                pht_predictions_train = exp1_data['outcomes'][len(cols_vec)][cost_func][0]['pht_predictions_train']
                pht_predictions_test = exp1_data['outcomes'][len(cols_vec)][cost_func][0]['pht_predictions_test']

                # calculate accuracies and append to df
                df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)

                df['Grassmann_dist'].append(None)

    return pd.DataFrame(df)

# Experiment 2 -----------------------------------

def plot_accuracy_time_exp2_opt_types(
    filename,
    winner_takes_all=True, 
    avg_type='macro',
    axis_label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    boxplot_linewidth=1.5,
    text_fontsize=10
):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename + '.pckl', 'rb')
    exp2_data = pickle.load(f)
    f.close()

    # load plot styles and extract
    general_styles = load_general_styles()

    # change to format easily passable to seaborn
    df = create_data_frame_exp2(exp2_data, winner_takes_all)

    # remove runs with bad training behaviour
    #df = df[(df.accuracy_train_macro > 0.9) | (df.dimension == 1)]

    # set time of dimension 3 to None
    df.loc[df['dimension'] == 3, 'time'] = None

    # Plot
    # create grouped boxplot
    column_width = general_styles['column_width']
    fig, ax1 = plt.subplots(figsize=(column_width, 0.6 * column_width))
    
    # Boxplot for accuracy
    box_acc = sns.boxplot(
        x=df['dimension'], 
        y=df['accuracy_test_' + avg_type],
        hue=df['opt_mode'],
        hue_order=['batch', 'iterative'],
        palette=[general_styles['colours']['blue'], general_styles['colours']['orange']],
        legend=True,
        showfliers=False,
        ax=ax1,
        linewidth=boxplot_linewidth
    )
    ax1.set(
        xlabel='subspace dimension $d^{\prime}$', 
        ylabel='test accuracy',
        ylim=[0, 1.01]
    )
    ax1.xaxis.label.set_size(axis_label_fontsize)
    ax1.yaxis.label.set_size(axis_label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Boxplot for time
    ax2 = ax1.twinx()
    box_time = sns.boxplot(
        x=df['dimension'], 
        y=df['time'],
        hue=df['opt_mode'],
        hue_order=['batch', 'iterative'],
        palette=[general_styles['colours']['blue'], general_styles['colours']['orange']],
        legend=True,
        showfliers=False,
        fill=False,
        ax=ax2,
        linewidth=boxplot_linewidth
    )
    ax2.set(
        ylabel='time (s)',
        ylim=[0, 20]
    )
    ax2.yaxis.label.set_size(axis_label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Adjust legend entries
    handles1, previous_labels = ax1.get_legend_handles_labels()
    ax1.legend(loc=(0.64, 0.2), handles=handles1, labels=['a) acc. - batch', 'b) acc. - iterative'], fontsize=legend_fontsize)

    handles2, previous_labels = ax2.get_legend_handles_labels()
    ax2.legend(loc='lower right', handles=handles2, labels=['c) time - batch', 'd) time - iterative'], fontsize=legend_fontsize)

    
    # Add text to image
    shift_small = 0.12
    shift_large = 1

    ax1.text(-0.1 - shift_small, 0.7, 'a)', fontsize=text_fontsize)
    ax1.text(0.3 - shift_small, 0.7, 'b)', fontsize=text_fontsize)
    ax1.text(-0.1 - shift_small, 0.06, 'c)', fontsize=text_fontsize)
    ax1.text(0.3 - shift_small, 0.06, 'd)', fontsize=text_fontsize)
    
    ax1.text(-0.1 - shift_small + shift_large, 0.95, 'a)', fontsize=text_fontsize)
    ax1.text(0.3 - shift_small + shift_large, 0.95, 'b)', fontsize=text_fontsize)
    ax1.text(-0.1 - shift_small + shift_large, 0.25, 'c)', fontsize=text_fontsize)
    ax1.text(0.3 - shift_small + shift_large, 0.15, 'd)', fontsize=text_fontsize)

    ax1.text(-0.1 - shift_small + 2 * shift_large, 0.95, 'a)', fontsize=text_fontsize)
    ax1.text(0.3 - shift_small + 2 * shift_large, 0.95, 'b)', fontsize=text_fontsize)

    #fig.suptitle(title_string)
    fig.tight_layout()
    plt.show()

    # print stats. on run time
    df_batch_dim1 = df[(df['opt_mode'].str.contains('batch')) & (df['dimension']==1)]['time']
    df_batch_dim2 = df[(df['opt_mode'].str.contains('batch')) & (df['dimension']==2)]['time']

    df_iter_dim1 = df[(df['opt_mode'].str.contains('iterative')) & (df['dimension']==1)]['time']
    df_iter_dim2 = df[(df['opt_mode'].str.contains('iterative')) & (df['dimension']==2)]['time']

    print('Batch, dim=1: \n', df_batch_dim1.describe(), '\n', '------------------------------------')
    print('Batch, dim=2: \n', df_batch_dim2.describe(), '\n', '------------------------------------')
    print('Iterative, dim=1: \n', df_iter_dim1.describe(), '\n', '------------------------------------')
    print('Iterative, dim=2: \n', df_iter_dim2.describe(), '\n', '------------------------------------')
    

    return fig

def create_data_frame_exp2(exp2_data, winner_takes_all = True):
    '''unrolls information stored in exp2_data into a dataframe-like format that is easily readable by Seaborn'''

    # make dict in dataframe format
    df = {}
    df['dimension'] = []
    df['opt_mode'] = []
    df['init_type'] = []

    df['time'] = []
    df['accuracy_train_micro'] = []
    df['accuracy_train_macro'] = []
    df['accuracy_test_micro'] = []
    df['accuracy_test_macro'] = []
    df['log_likelihood_train'] = []
    df['log_likelihood_test'] = []
    df['BIC'] = []
    df['AIC'] = []

    # unpack from exp1_data
    train_index = exp2_data['train_index']
    test_index = exp2_data['test_index']
    
    data_box_train = exp2_data['data_box_spherical'].select_examples(train_index)
    data_box_test = exp2_data['data_box_spherical'].select_examples(test_index)

    labels_train = data_box_train.get_labels()
    labels_test = data_box_test.get_labels()
    
    # extract experimental info
    cols_vec = exp2_data['exp2_opts']['cols_vec']    
    num_rand_init = exp2_data['exp2_opts']['num_rand_init']
    num_dim_ambient = exp2_data['data_box_spherical'].get_dynamical_system().num_params

    # loop over dimensionalities
    for cols_idx in range(len(cols_vec)):

        # loop over cost function types
        for opt_mode in ['batch', 'iterative']:

            # loop over random inits
            for trial in range(num_rand_init):
                df['dimension'].append(cols_vec[cols_idx])
                df['opt_mode'].append(opt_mode)
                df['init_type'].append('random')
                
                df['time'].append(exp2_data['outcomes'][cols_idx][opt_mode][trial]['ssl_time'])
                
                # extract predictions and evaluate accuracy
                pht_predictions_train = exp2_data['outcomes'][cols_idx][opt_mode][trial]['pht_predictions_train']
                pht_predictions_test = exp2_data['outcomes'][cols_idx][opt_mode][trial]['pht_predictions_test']

                # calculate accuracies and append to df
                df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)
                
            # add GMLVQ init
            df['dimension'].append(cols_vec[cols_idx])
            df['opt_mode'].append(opt_mode)
            df['init_type'].append('GMLVQ')
            
            df['time'].append(exp2_data['outcomes'][cols_idx][opt_mode][num_rand_init]['ssl_time'])
            
            # extract predictions and evaluate accuracy
            pht_predictions_train = exp2_data['outcomes'][cols_idx][opt_mode][num_rand_init]['pht_predictions_train']
            pht_predictions_test = exp2_data['outcomes'][cols_idx][opt_mode][num_rand_init]['pht_predictions_test']
            
            # calculate accuracies and append to df
            df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)
    
    # add results for ambient dimension
    for opt_mode in ['batch', 'iterative']:
        for init_type in ['random', 'GMLVQ']:
    
            df['dimension'].append(num_dim_ambient)
            df['opt_mode'].append(opt_mode)
            df['init_type'].append(init_type)
            
            df['time'].append(exp2_data['outcomes'][len(cols_vec)][opt_mode][0]['ssl_time'])

            # extract predictions and evaluate accuracy
            pht_predictions_train = exp2_data['outcomes'][len(cols_vec)][opt_mode][0]['pht_predictions_train']
            pht_predictions_test = exp2_data['outcomes'][len(cols_vec)][opt_mode][0]['pht_predictions_test']

            # calculate accuracies and append to df
            df = append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, cols_vec[cols_idx], winner_takes_all)

    return pd.DataFrame(df)

# Experiment 3 -> did not make it into the paper :(

# Experiment 4a -> did not make it into the paper :(

# Experiment 4b ----------------------------------

def boxplot_accuracy_loglikelihood_exp4b(
        filename,
        winner_takes_all = True,
        avg_type = 'macro',
        axis_label_fontsize = 12,
        tick_fontsize = 10,
        legend_fontsize = 10,
        boxplot_linewidth = 1.5
        ):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename + '.pckl', 'rb')
    exp4b_data = pickle.load(f)
    f.close()

    # set title string
    title_string = 'Gravitational Waves (winner_takes_all: ' + str(winner_takes_all) + ' | ' + avg_type + ' averages)'

    # change to format easily passable to seaborn
    df = create_data_frame_exp4b(exp4b_data, winner_takes_all = winner_takes_all)

    # add indicator columns for train / test performance
    df_train = deepcopy(df)
    df_train['accuracy'] = df_train['accuracy_train_' + avg_type]
    df_train['log_likelihood'] = df_train['log_likelihood_train']
    df_train['train/test'] = 'train'

    df_test = deepcopy(df)
    df_test['accuracy'] = df_test['accuracy_test_' + avg_type]
    df_test['log_likelihood'] = df_train['log_likelihood_test']
    df_test['train/test'] = 'test'
    
    # combine in single dataframe
    df_long = pd.concat([df_train, df_test], ignore_index=True, axis=0)

    # load plot styles and extract
    general_styles= load_general_styles()
    
    # create grouped boxplot
    #df_random = df.loc[df['init_type'] == 'random']

    # Plot
    column_width = general_styles['column_width']
    #fig, ax = plt.subplots(figsize=(2 * column_width, (2 * column_width) / 4))
    fig, ax1 = plt.subplots(figsize=(column_width, 0.6 * column_width))

    ax1 = sns.boxplot(
        x = df_long['dimension'], 
        y = df_long['accuracy'],
        hue = df_long['train/test'],
        palette = [general_styles['colours']['blue'], general_styles['colours']['orange']],
        showmeans=False,
        showfliers=False,
        linewidth=boxplot_linewidth,
        ax=ax1
        )
    ax1.set(
                xlabel = 'subspace dimension $d^{\prime}$', 
                ylabel = 'accuracy',
                #title = 'Accuracy',
                ylim = [0, 1.01]
            )
    ax1.xaxis.label.set_size(axis_label_fontsize)
    ax1.yaxis.label.set_size(axis_label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    handles1, previous_labels = ax1.get_legend_handles_labels()
    ax1.legend(loc=(0.45, 0.019), 
                handles=handles1, 
                labels=['train acc.', 'test acc.'], 
                fontsize=legend_fontsize)

    # Boxplot for time
    ax2 = ax1.twinx()
    box_time = sns.boxplot(
        x = df_long['dimension'], 
        y = df_long['log_likelihood'],
        hue = df_long['train/test'],
        palette = [general_styles['colours']['blue'], general_styles['colours']['orange']],
        showmeans=False,
        showfliers=False,
        fill=False,
        linewidth=boxplot_linewidth,
        ax=ax2
    )
    ax2.set(
                xlabel = 'subspace dimension $d^{\prime}$', 
                ylabel = 'log-likelihood',
                #title = 'Accuracy',
                ylim = [-25, 20]
            )
    ax2.xaxis.label.set_size(axis_label_fontsize)
    ax2.yaxis.label.set_size(axis_label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    handles2, previous_labels = ax2.get_legend_handles_labels()
    ax2.legend(loc='lower right', 
               handles=handles2, 
               labels=['train log-like.', 'test log-like.'], 
               fontsize=legend_fontsize)

    #fig.suptitle(title_string)
    fig.tight_layout()
    plt.show()

    return fig

def create_data_frame_exp4b(exp4b_data, winner_takes_all = True):
    '''unrolls information stored in exp4_data into a dataframe-like format that is easily readable by Seaborn'''

    # make dict in dataframe format
    df = {}

    df['fold'] = []
    df['dimension'] = []
    df['init_type'] = []
    df['time'] = []

    df['accuracy_train_micro'] = []
    df['accuracy_train_macro'] = []
    df['accuracy_test_micro'] = []
    df['accuracy_test_macro'] = []
    df['log_likelihood_train'] = []
    df['log_likelihood_test'] = []
    df['BIC'] = []
    df['AIC'] = []
    
    # extract experimental info
    num_splits = exp4b_data['exp4b_opts']['num_splits']
    cols_vec = exp4b_data['exp4b_opts']['cols_vec']    
    num_rand_init = exp4b_data['exp4b_opts']['num_rand_init']
    num_dim_ambient = exp4b_data['data_box_spherical'].get_dynamical_system().num_params

    for fold_idx in range(num_splits):

        # unpack from exp4_data
        train_index = exp4b_data['outcomes_folds'][fold_idx]['train_index']
        test_index = exp4b_data['outcomes_folds'][fold_idx]['test_index']
        
        data_box_train = exp4b_data['data_box_spherical'].select_examples(train_index)
        data_box_test = exp4b_data['data_box_spherical'].select_examples(test_index)

        labels_train = data_box_train.get_labels()
        labels_test = data_box_test.get_labels()

        # loop over dimensionalities
        for cols_idx in range(len(cols_vec)):

            # loop over cost function types
            for init_type in ['random_init', 'GMLVQ_init']:

                # loop over initializations
                for trial in range(num_rand_init):
                    
                    df['fold'].append(fold_idx + 1)
                    df['dimension'].append(cols_vec[cols_idx])
                    df['init_type'].append(init_type)
                    
                    df['time'].append(exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][cols_idx][init_type][trial]['ssl_time'])

                    # extract predictions
                    pht_predictions_train = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][cols_idx][init_type][trial]['pht_predictions_train']
                    pht_predictions_test = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][cols_idx][init_type][trial]['pht_predictions_test']
                    
                    # evaluate micro and macro averaged accuracy
                    acc_train_micro, acc_train_macro = utility.calc_accuracy(pht_predictions_train, labels_train, winner_takes_all)
                    acc_test_micro, acc_test_macro = utility.calc_accuracy(pht_predictions_test, labels_test, winner_takes_all)

                    # record in dataframe
                    df['accuracy_train_micro'].append(acc_train_micro)
                    df['accuracy_train_macro'].append(acc_train_macro)
                    df['accuracy_test_micro'].append(acc_test_micro)
                    df['accuracy_test_macro'].append(acc_test_macro)

                    # calculate log-likelihood, BIC and AIC
                    log_likelihood_train = utility.calc_log_likelihood(pht_predictions_train, labels_train)
                    log_likelihood_test = utility.calc_log_likelihood(pht_predictions_test, labels_test)
                    df['log_likelihood_train'].append(log_likelihood_train)
                    df['log_likelihood_test'].append(log_likelihood_test)
                    df['BIC'].append(utility.calc_BIC(cols_vec[cols_idx], len(labels_test), log_likelihood_test))
                    df['AIC'].append(utility.calc_AIC(cols_vec[cols_idx], log_likelihood_test))
                    
        # add results for ambient dimension
        for init_type in ['random_init', 'GMLVQ_init']:
    
            df['fold'].append(fold_idx + 1)
            df['dimension'].append(num_dim_ambient)
            df['init_type'].append(init_type)
            
            df['time'].append(exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][len(cols_vec)][init_type][0]['ssl_time'])
            #df['accuracy'].append(exp4_data['outcomes_folds'][fold_idx][len(cols_vec)][cost_func][0]['accuracy'])

            # extract predictions and evaluate accuracy
            pht_predictions_train = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][len(cols_vec)][init_type][0]['pht_predictions_train']
            pht_predictions_test = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][len(cols_vec)][init_type][0]['pht_predictions_test']

            # evaluate micro and macro averaged accuracy
            acc_train_micro, acc_train_macro = utility.calc_accuracy(pht_predictions_train, labels_train, winner_takes_all)
            acc_test_micro, acc_test_macro = utility.calc_accuracy(pht_predictions_test, labels_test, winner_takes_all)

            # record in dataframe
            df['accuracy_train_micro'].append(acc_train_micro)
            df['accuracy_train_macro'].append(acc_train_macro)
            df['accuracy_test_micro'].append(acc_test_micro)
            df['accuracy_test_macro'].append(acc_test_macro)

            # calculate log-likelihood, BIC and AIC
            log_likelihood_train = utility.calc_log_likelihood(pht_predictions_train, labels_train)
            log_likelihood_test = utility.calc_log_likelihood(pht_predictions_test, labels_test)
            df['log_likelihood_train'].append(log_likelihood_train)
            df['log_likelihood_test'].append(log_likelihood_test)
            df['BIC'].append(utility.calc_BIC(cols_vec[cols_idx], len(labels_test), log_likelihood_test))
            df['AIC'].append(utility.calc_AIC(cols_vec[cols_idx], log_likelihood_test))

    return pd.DataFrame(df)

def plot_projected_densities_4b_relevance_profile(
        filename, 
        winner_takes_all = True, 
        avg_type = 'macro', 
        plot_opts = None,
        title_fontsize = 16,
        axis_label_fontsize=12,
        tick_fontsize=10,
        legend_fontsize=10,
        linewidth=3,
        heatmap_fontsize = 10
        ):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename + '.pckl', 'rb')
    exp4b_data = pickle.load(f)
    f.close()

    # extract data box
    data_box_spherical = exp4b_data['data_box_spherical']

    # extract experimental info
    num_folds = exp4b_data['exp4b_opts']['num_splits'] 

    # initialize list to hold all projection matrices
    linear_map_mat_lists = [[], []]
    projection_mat_lists = [[], []]

    # loop over dimensions
    for dim in [1, 2]:

        # loop over folds
        for fold_idx in range(num_folds):

            # unpack from exp4b_data
            test_index = exp4b_data['outcomes_folds'][fold_idx]['test_index']
            data_box_test = exp4b_data['data_box_spherical'].select_examples(test_index)
            labels_test = data_box_test.get_labels()

            # loop over initialization type
            for init_type in ['random_init', 'GMLVQ_init']:
                
                # loop over number of initializations
                num_inits = len(exp4b_data['outcomes_folds'][0]['dimension_results'][0][init_type])
                for init_idx in range(num_inits):

                    # extract learned subspace
                    V_opt = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][dim-1][init_type][init_idx]['subspace_learned']['V_opt']

                    # extract predictions
                    pht_predictions_test = exp4b_data['outcomes_folds'][fold_idx]['dimension_results'][dim-1][init_type][init_idx]['pht_predictions_test']

                    # check which accuracy is wanted
                    acc_micro, acc_macro = utility.calc_accuracy(pht_predictions_test, labels_test, winner_takes_all = winner_takes_all)
                    if avg_type == 'macro':
                        accuracy = acc_macro
                    else:
                        accuracy = acc_micro

                    # add canonical projection to list
                    linear_map_mat_lists[dim-1].append(V_opt)
                    projection_mat_lists[dim-1].append(np.dot(V_opt, V_opt.T))
    
    # --- DIMENSION d'=1 and d' = 2 ---
    angles_list = []

    identity_mat = np.identity(4)

    # loop over dimensions
    for dim in [0, 1]:
        
        # initialize array to hold angles
        angles_mat = np.zeros((len(linear_map_mat_lists[dim]), 4))

        # loop over learned matrices
        for i in range(len(linear_map_mat_lists[dim])):

            # loop over mass_1, mass_2, chi_1 and chi_2
            V = linear_map_mat_lists[dim][i]
            for par in range(4):
                
                # define basis vector and project onto span(V)
                basis_vec = identity_mat[:, par]
                proj_vector = np.dot(np.dot(V, V.T), basis_vec)

                # calculate angle between basis vector and its projection
                angles_mat[i, par] = np.pi / 2 - np.arccos(np.dot(basis_vec, proj_vector) / (np.linalg.norm(basis_vec) * np.linalg.norm(proj_vector)))
        
        # append matrix containing angles
        angles_list.append(angles_mat)

    # Plot

    # load styles
    gw_styles = load_gravitational_waves_styles()
    class_labels = gw_styles['class_labels']
    class_colours = gw_styles['class_colours']
    general_styles = load_general_styles()

    # create figure
    column_width = general_styles['column_width']
    fig, axs = plt.subplots(1, 4, figsize=(2 * column_width, (2 * column_width) / 4))

    # --- Relevance matrix d'=1 ---

    sns.barplot(
        angles_list[0],
        ax=axs[0],
        color='tab:grey'
        )
    
    axs[0].set_title('a) avg. 1D relevance profile', fontsize=title_fontsize)
    axs[0].set_xticks(ticks = [0, 1, 2, 3], labels = [r'$m_{1}$', r'$m_{2}$', r'$\chi_{1}$', r'$\chi_{2}$'])
    axs[0].set_ylabel(r'$\pi / 2 - \beta$')
    axs[0].set_yticks(ticks = [0, np.pi / 4, np.pi / 2], labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    axs[0].xaxis.label.set_size(axis_label_fontsize)
    axs[0].yaxis.label.set_size(axis_label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # --- Relevance matrix d'=2 ---

    sns.barplot(
        angles_list[1],
        ax=axs[1],
        color='tab:grey'
        )
    
    axs[1].set_title('b) avg. 2D relevance profile', fontsize=title_fontsize)
    axs[1].set_xticks(ticks = [0, 1, 2, 3], labels = [r'$m_{1}$', r'$m_{2}$', r'$\chi_{1}$', r'$\chi_{2}$'])
    axs[1].set_ylabel(r'$\pi / 2 - \beta$')
    axs[1].set_yticks(ticks = [0, np.pi / 4, np.pi / 2], labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    axs[1].xaxis.label.set_size(axis_label_fontsize)
    axs[1].yaxis.label.set_size(axis_label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)


    # --- Projections d'=1 ---

    # find average projection mat
    projection_mat_array_1 = np.dstack(projection_mat_lists[0])
    mean_projection_mat_1 = np.mean(projection_mat_array_1, axis=2)

    # extract from data box
    data_box_projected_1 = data_box_spherical.project_density_estimates(mean_projection_mat_1[:, :1] / np.linalg.norm(mean_projection_mat_1[:, 0]))
    labels = data_box_projected_1.get_labels()
    num_examples = len(labels)
    density_estimates = data_box_projected_1.get_density_estimates()
    class_flag = np.zeros(len(class_labels))
    for n in range(num_examples):

        # plot Gaussian mixture
        num_comps = len(density_estimates[n]['mix_weights'])
        f = np.zeros(plot_opts['x_d1'].shape)
        for k in range(num_comps):

            # evaluate Gaussian on x
            mu = density_estimates[n]['mu_array'][:, k]
            Sigma = density_estimates[n]['Sigma_array'][k, :, :]
            weight = density_estimates[n]['mix_weights'][k]
            y = multivariate_normal.pdf(plot_opts['x_d1'], mu, Sigma)

            # accumulate from different components
            f += weight * y
        
        # plot Gaussian mixture in the class colour
        if class_flag[labels[n]] == 0:
            axs[2].plot(plot_opts['x_d1'], f, color=class_colours[labels[n]], label=class_labels[labels[n]], alpha=plot_opts['alpha'], linewidth=linewidth)
            class_flag[labels[n]] = 1
        else:
            axs[2].plot(plot_opts['x_d1'], f, color=class_colours[labels[n]], alpha=plot_opts['alpha'], linewidth=linewidth)
    
    axs[2].set(
        xlabel = r'$\tilde{v}_{1} \cdot [m_{1}, m_{2}, \chi_{1}, \chi_{2}]$',
        ylabel = 'density'
    )
    axs[2].set_title('c) 1D projection of mixtures', fontsize=title_fontsize)
    axs[2].xaxis.label.set_size(axis_label_fontsize)
    axs[2].yaxis.label.set_size(axis_label_fontsize)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Adjust legend entries
    handles1, previous_labels = axs[2].get_legend_handles_labels()
    axs[2].legend(loc='upper right', handles=handles1, labels=['BNS/NSBH', 'Small BBH', 'Large BBH'], fontsize=legend_fontsize)


    # --- Projections d'=2 ---

    # find average projection mat
    projection_mat_array_2 = np.dstack(projection_mat_lists[1])
    mean_projection_mat_2 = np.mean(projection_mat_array_2, axis=2)

    # extract from data box
    data_box_projected_2 = data_box_spherical.project_density_estimates(np.dot(mean_projection_mat_2[:, :2], np.diag(1 / np.linalg.norm(mean_projection_mat_2[:, :2], axis=0))))
    labels = data_box_projected_2.get_labels().astype(int)
    num_examples = len(labels)
    density_estimates = data_box_projected_2.get_density_estimates()
    
    # create mesh-grid
    X, Y = np.meshgrid(plot_opts['x_d2'], plot_opts['y_d2'])
    pos = np.dstack((X, Y))

    # initalize class conditional arrays
    Z_C0 = np.zeros(X.shape)
    Z_C1 = np.zeros(X.shape)
    Z_C2 = np.zeros(X.shape)

    class_flag = np.zeros(len(class_labels))
    for n in range(num_examples):

        # plot Gaussian mixture
        num_comps = len(density_estimates[n]['mix_weights'])
        Z = np.zeros(X.shape)
        for k in range(num_comps):

            # evaluate Gaussian on x
            mu = density_estimates[n]['mu_array'][:, k]
            Sigma = density_estimates[n]['Sigma_array'][k, :, :]
            weight = density_estimates[n]['mix_weights'][k]
            Z_add = multivariate_normal.pdf(pos, mu, Sigma)

            # accumulate from different components
            Z += weight * Z_add

        # add Z to class conditional arrays
        if labels[n] == 0:
            Z_C0 += Z
        elif labels[n] == 1:
            Z_C1 += Z
        else:
            Z_C2 += Z

        # find maximum density value
        vmax = np.max(Z)
        vmin = np.min(Z)
        levels = np.linspace(0.1 * vmax, vmax, num=10)

        # plot Gaussian mixture in the class colour
        if class_flag[labels[n]] == 0:
            axs[3].contour(X, Y, Z, levels=levels, colors=class_colours[labels[n]], linewidths=linewidth)
            class_flag[labels[n]] = 1
        else:
            axs[3].contour(X, Y, Z, levels=levels, colors=class_colours[labels[n]], alpha = plot_opts['alpha'], linewidths=linewidth)
    
    # add invisible plots for labels
    axs[3].plot(-plot_opts['x_d1'], -plot_opts['x_d1'], color=class_colours[0], alpha = plot_opts['alpha'], label='BNS/NSBH', linewidth=linewidth)
    axs[3].plot(-plot_opts['x_d1'], -plot_opts['x_d1'], color=class_colours[1], alpha = plot_opts['alpha'], label='Small BBH', linewidth=linewidth)
    axs[3].plot(-plot_opts['x_d1'], -plot_opts['x_d1'], color=class_colours[2], alpha = plot_opts['alpha'], label='Large BBH', linewidth=linewidth) 

    axs[3].set(
        xlabel = r'$\tilde{w}_{1} \cdot [m_{1}, m_{2}, \chi_{1}, \chi_{2}]$',
        ylabel = r'$\tilde{w}_{2} \cdot [m_{1}, m_{2}, \chi_{1}, \chi_{2}]$'
    )
    axs[3].set_xlim([np.min(plot_opts['x_d2']), np.max(plot_opts['x_d2'])])
    axs[3].set_ylim([np.min(plot_opts['y_d2']), np.max(plot_opts['y_d2'])])
    axs[3].set_title('d) 2D projection of mixtures', fontsize=title_fontsize)
    axs[3].xaxis.label.set_size(axis_label_fontsize)
    axs[3].yaxis.label.set_size(axis_label_fontsize)
    axs[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[3].legend(loc='upper left', fontsize=legend_fontsize)

    fig.tight_layout()

    return fig

# Experiment 5 -----------------------------------

def plot_optimization_history_averages(
    filename, 
    winner_takes_all=True, 
    num_std=3, 
    axis_label_fontsize=12, 
    tick_fontsize=10, 
    legend_fontsize=10, 
    line_width=2
):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename + '.pckl', 'rb')
    exp5_data = pickle.load(f)
    f.close()

    # extract outcomes from loaded data
    outcomes = exp5_data['outcomes']
    num_orth_init = len(outcomes)
    data_box_spherical = exp5_data['data_box_spherical']
    subspace = data_box_spherical.get_subspace()
    V_true = np.vstack((subspace['v1'], subspace['v2'])).T
    normal_vec = -np.cross(subspace['v1'], subspace['v2'])

    labels_train = data_box_spherical.get_labels()[exp5_data['train_index']]
    labels_test = data_box_spherical.get_labels()[exp5_data['test_index']]

    # find maximum number of iterations
    max_num_iterations = 0
    for i in range(num_orth_init):
        num_iterations = len(outcomes[i]['subspace_learned']['opt_history']['fun'])
        if num_iterations > max_num_iterations:
            max_num_iterations = num_iterations

    # initialize arrays
    loglikelihood_values = np.zeros((num_orth_init, max_num_iterations))
    Grassmann_distances = np.zeros((num_orth_init, max_num_iterations))
    train_accuracies = np.zeros((num_orth_init, max_num_iterations))
    test_accuracies = np.zeros((num_orth_init, max_num_iterations))
    normals_angle = np.zeros((num_orth_init, max_num_iterations))

    # loop over subspace initalizations
    for i in range(num_orth_init):
        
        # find number of iterations for current optimization
        num_iterations = len(outcomes[i]['subspace_learned']['opt_history']['fun'])
        
        # loop over iterations
        for it in range(max_num_iterations):
            
            # log-likelihood
            if it == 0:
                loglikelihood_values[i, it] = outcomes[i]['subspace_learned']['fun0']
            elif it < num_iterations:
                loglikelihood_values[i, it] = outcomes[i]['subspace_learned']['opt_history']['fun'][it]
            else:
                loglikelihood_values[i, it] = outcomes[i]['subspace_learned']['opt_history']['fun'][num_iterations - 1]

            # Grassmann distances
            if it == 0:
                Grassmann_distances[i, it] = utility.GrassmannDist(V_true, outcomes[i]['subspace_learned']['V0'])
            elif it < num_iterations:
                Grassmann_distances[i, it] = utility.GrassmannDist(V_true, outcomes[i]['subspace_learned']['opt_history']['V'][it])
            else:
                Grassmann_distances[i, it] = utility.GrassmannDist(V_true, outcomes[i]['subspace_learned']['opt_history']['V'][num_iterations - 1])

            # angle between normal vectors
            if it == 0:
                V_it = outcomes[i]['subspace_learned']['V0']
                normal_vec_it = np.cross(V_it[:, 0], V_it[:, 1])
                normals_angle[i, it] = np.arccos(np.dot(normal_vec, normal_vec_it) / (np.linalg.norm(normal_vec) * np.linalg.norm(normal_vec_it)))
            elif it < num_iterations:
                V_it = outcomes[i]['subspace_learned']['opt_history']['V'][it]
                normal_vec_it = np.cross(V_it[:, 0], V_it[:, 1])
                normals_angle[i, it] = np.arccos(np.dot(normal_vec, normal_vec_it) / (np.linalg.norm(normal_vec) * np.linalg.norm(normal_vec_it)))
            else:
                V_it = outcomes[i]['subspace_learned']['opt_history']['V'][num_iterations - 1]
                normal_vec_it = np.cross(V_it[:, 0], V_it[:, 1])
                normals_angle[i, it] = np.arccos(np.dot(normal_vec, normal_vec_it) / (np.linalg.norm(normal_vec) * np.linalg.norm(normal_vec_it)))

            # Train and test accuracy
            if it < num_iterations + 1:
                _, macro_acc_train = utility.calc_accuracy(outcomes[i]['predictions'][it]['train'], labels_train, winner_takes_all=winner_takes_all)
                _, macro_acc_test = utility.calc_accuracy(outcomes[i]['predictions'][it]['test'], labels_test, winner_takes_all=winner_takes_all)
                train_accuracies[i, it] = macro_acc_train
                test_accuracies[i, it] = macro_acc_test
            else:
                _, macro_acc_train = utility.calc_accuracy(outcomes[i]['predictions'][num_iterations]['train'], labels_train, winner_takes_all=winner_takes_all)
                _, macro_acc_test = utility.calc_accuracy(outcomes[i]['predictions'][num_iterations]['test'], labels_test, winner_takes_all=winner_takes_all)
                train_accuracies[i, it] = macro_acc_train
                test_accuracies[i, it] = macro_acc_test

    # plotting
    # load general styles
    general_styles = load_general_styles()
    column_width = general_styles['column_width']
    fig, axs = plt.subplots(2, 2, figsize=(column_width, 0.6 * column_width))

    # loglikelihood
    loglikelihood_values = -loglikelihood_values
    loglikelihood_mean = np.mean(loglikelihood_values, axis=0)
    loglikelihood_std = np.std(loglikelihood_values, axis=0)

    # Grassmann_distances
    Grassmann_mean = np.mean(Grassmann_distances, axis=0)
    Grassmann_std = np.std(Grassmann_distances, axis=0)

    # normals angle
    normals_mean = np.mean(normals_angle, axis=0)
    normals_std = np.std(normals_angle, axis=0)

    # train_accuracies
    train_accuracies_mean = np.mean(train_accuracies, axis=0)
    train_accuracies_std = np.std(train_accuracies, axis=0)

    # test_accuracies
    test_accuracies_mean = np.mean(test_accuracies, axis=0)
    test_accuracies_std = np.std(test_accuracies, axis=0)

    # plots with adjustable line width
    axs[0, 0].plot(loglikelihood_mean, 'k', label='mean', linewidth=line_width)
    axs[0, 0].fill_between(np.arange(max_num_iterations), loglikelihood_mean - num_std * loglikelihood_std, loglikelihood_mean + num_std * loglikelihood_std, color='k', alpha=0.2, label=str(num_std) + '-std.')
    axs[0, 0].legend(loc = 'lower right', fontsize=legend_fontsize)

    axs[0, 1].plot(normals_mean, 'k', label='mean', linewidth=line_width)
    axs[0, 1].fill_between(np.arange(max_num_iterations), normals_mean - num_std * normals_std, normals_mean + num_std * normals_std, color='k', alpha=0.2, label=str(num_std) + '-std.')
    axs[0, 1].legend(loc = 'upper right',fontsize=legend_fontsize)

    axs[1, 0].plot(train_accuracies_mean, 'k', label='mean', linewidth=line_width)
    axs[1, 0].fill_between(np.arange(max_num_iterations), train_accuracies_mean - num_std * train_accuracies_std, train_accuracies_mean + num_std * train_accuracies_std, color='k', alpha=0.2, label=str(num_std) + '-std.')
    axs[1, 0].legend(loc = 'lower right',fontsize=legend_fontsize)

    axs[1, 1].plot(test_accuracies_mean, 'k', label='mean', linewidth=line_width)
    axs[1, 1].fill_between(np.arange(max_num_iterations), test_accuracies_mean - num_std * test_accuracies_std, test_accuracies_mean + num_std * test_accuracies_std, color='k', alpha=0.2, label=str(num_std) + '-std.')
    axs[1, 1].legend(loc = 'lower right',fontsize=legend_fontsize)

    # ddjust label sizes
    for i in [0, 1]:
        for j in [0, 1]:
            axs[i, j].tick_params(axis='both', labelsize=tick_fontsize)

    # set axis configs
    axs[0, 0].set_xlabel('iteration', fontsize=axis_label_fontsize)
    axs[0, 0].set_ylabel('log-likelihood', fontsize=axis_label_fontsize)
    
    axs[0, 1].set_xlabel('iteration', fontsize=axis_label_fontsize)
    axs[0, 1].set_ylabel('normals angle', fontsize=axis_label_fontsize)
    axs[0, 1].set_ylim([0, np.pi / 2])
    axs[0, 1].set_yticks([0, np.pi / 4, np.pi / 2])
    axs[0, 1].set_yticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])

    axs[1, 0].set_ylim([0.7, 1])
    axs[1, 0].set_yticks([0.7, 0.8, 0.9, 1])
    axs[1, 0].set_xlabel('iteration', fontsize=axis_label_fontsize)
    axs[1, 0].set_ylabel('train acc.', fontsize=axis_label_fontsize)

    axs[1, 1].set_ylim([0.7, 1])
    axs[1, 1].set_yticks([0.7, 0.8, 0.9, 1])
    axs[1, 1].set_xlabel('iteration', fontsize=axis_label_fontsize)
    axs[1, 1].set_ylabel('test acc.', fontsize=axis_label_fontsize)

    
    fig.tight_layout()

    return fig

# Experiment 6 -----------------------------------

def plot_GMLVQ_SSL_comparisson(
        filename, 
        winner_takes_all = True,
        axis_label_fontsize=12,
        tick_fontsize=10,
        legend_fontsize=10,
        boxplot_linewidth=1.5
        ):

    # load experiment outcomes from file
    f = open('../data/processed/' + filename + '.pckl', 'rb')
    exp6_data = pickle.load(f)
    f.close()

    # extract outcomes from loaded data
    outcomes = exp6_data['outcomes']
    data_box_spherical = exp6_data['data_box_spherical']
    num_rand_init = exp6_data['exp6_opts']['num_rand_init']
    
    labels_train = data_box_spherical.get_labels()[exp6_data['train_index']]
    labels_test = data_box_spherical.get_labels()[exp6_data['test_index']]

    # make dict in dataframe format
    df = {}
    df['method'] = []
    df['train/test'] = []
    df['accuracy'] = []

    for i in range(num_rand_init):

        # GMLVQ
        _, macro_acc_train = utility.calc_accuracy(outcomes[i]['GMLVQ_predictions_train'], outcomes[i]['GMLVQ_labels_train'], winner_takes_all = winner_takes_all)
        _, macro_acc_test = utility.calc_accuracy(outcomes[i]['GMLVQ_predictions_test'], outcomes[i]['GMLVQ_labels_test'], winner_takes_all = winner_takes_all)
        
        df['method'].append('GMLVQ')
        df['method'].append('GMLVQ')
        df['train/test'].append('train')
        df['train/test'].append('test')
        df['accuracy'].append(macro_acc_train)
        df['accuracy'].append(macro_acc_test)

        # PHT evaluated on subspace learned by GMLVQ
        _, macro_acc_train = utility.calc_accuracy(outcomes[i]['pht_predictions_train_GMLVQ'], labels_train, winner_takes_all = winner_takes_all)
        _, macro_acc_test = utility.calc_accuracy(outcomes[i]['pht_predictions_test_GMLVQ'], labels_test, winner_takes_all = winner_takes_all)
        
        df['method'].append('PHT')
        df['method'].append('PHT')
        df['train/test'].append('train')
        df['train/test'].append('test')
        df['accuracy'].append(macro_acc_train)
        df['accuracy'].append(macro_acc_test)

        # PHT evaluated on subspace learned by ssl
        _, macro_acc_train = utility.calc_accuracy(outcomes[i]['pht_predictions_train'], labels_train, winner_takes_all = winner_takes_all)
        _, macro_acc_test = utility.calc_accuracy(outcomes[i]['pht_predictions_test'], labels_test, winner_takes_all = winner_takes_all)

        df['method'].append('PHT_SSL')
        df['method'].append('PHT_SSL')
        df['train/test'].append('train')
        df['train/test'].append('test')
        df['accuracy'].append(macro_acc_train)
        df['accuracy'].append(macro_acc_test)

    # PHT evaluated on subspace learned by ssl
    _, macro_acc_train = utility.calc_accuracy(outcomes[num_rand_init]['pht_predictions_train'], labels_train, winner_takes_all = winner_takes_all)
    _, macro_acc_test = utility.calc_accuracy(outcomes[num_rand_init]['pht_predictions_test'], labels_test, winner_takes_all = winner_takes_all)

    df['method'].append('PHT_ambient')
    df['method'].append('PHT_ambient')
    df['train/test'].append('train')
    df['train/test'].append('test')
    df['accuracy'].append(macro_acc_train)
    df['accuracy'].append(macro_acc_test)

    # Plot

    # load plot styles and extract
    general_styles = load_general_styles()
    column_width = general_styles['column_width']
    fig, axis = plt.subplots(figsize=(column_width, 0.6 * column_width))
    ax = sns.boxplot(
                data = df,
                x = 'method',
                y = 'accuracy',
                hue = 'train/test',
                showmeans=False,
                showfliers=True,
                ax=axis,
                linewidth=boxplot_linewidth)
    ax.set(
        xlabel='', 
        ylabel='accuracy',
        ylim=[0.6, 1.01]
    )
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xticks([0, 1, 2, 3], labels=['GMLVQ', 'PHT', 'PHT + SSL', 'PHT in 3D'])
    ax.legend(loc='lower right', fontsize=legend_fontsize)

    return fig

# General ----------------------------------------
def print_meta_data(filename):

    if 'exp4_' in filename:

        # load experiment outcomes from file
        f = open('../data/processed/' + filename + '.pckl', 'rb')
        exp4_data = pickle.load(f)
        f.close()

        print('Filename:            ' + filename)
        print('Number of examples:  ' + str(len(exp4_data['data_box_spherical'].get_labels())))
        print('Total time (h):      ' + str(np.round(exp4_data['total_time'] / 3600, 2)))
        print('Folds intended:      ' + str(exp4_data['exp4_opts']['num_folds']))
        print('Folds completed:     ' + str(exp4_data['folds_completed']))

    elif 'exp4b' in filename:

        # load experiment outcomes from file
        f = open('../data/processed/' + filename + '.pckl', 'rb')
        exp4b_data = pickle.load(f)
        f.close()

        print('Filename:             ' + filename)
        print('Number of examples:   ' + str(len(exp4b_data['data_box_spherical'].get_labels())))
        print('Total time (h):       ' + str(np.round(exp4b_data['total_time'] / 3600, 2)))
        print('Resamplings intended: ' + str(exp4b_data['exp4b_opts']['num_splits']))
        print('Folds completed:      ' + str(exp4b_data['folds_completed']))

    else:

        # load experiment outcomes from file
        f = open('../data/processed/' + filename + '.pckl', 'rb')
        exp_data = pickle.load(f)
        f.close()

        print('Filename:            ' + filename)
        print('Number of examples:  ' + str(len(exp_data['data_box_spherical'].get_labels())))
        print('Total time (h):      ' + str(np.round(exp_data['total_time'] / 3600, 2)))

def append_accuracies_to_df(df, pht_predictions_train, pht_predictions_test, labels_train, labels_test, num_parameters, winner_takes_all):
    
    # evaluate micro and macro averaged accuracy
    acc_train_micro, acc_train_macro = utility.calc_accuracy(pht_predictions_train, labels_train, winner_takes_all)
    acc_test_micro, acc_test_macro = utility.calc_accuracy(pht_predictions_test, labels_test, winner_takes_all)

    # record in dataframe
    df['accuracy_train_micro'].append(acc_train_micro)
    df['accuracy_train_macro'].append(acc_train_macro)
    df['accuracy_test_micro'].append(acc_test_micro)
    df['accuracy_test_macro'].append(acc_test_macro)

    # calculate log-likelihood, BIC and AIC
    log_likelihood_train = utility.calc_log_likelihood(pht_predictions_train, labels_train)
    log_likelihood_test = utility.calc_log_likelihood(pht_predictions_test, labels_test)
    df['log_likelihood_train'].append(log_likelihood_train)
    df['log_likelihood_test'].append(log_likelihood_test)
    df['BIC'].append(utility.calc_BIC(num_parameters, len(labels_test), log_likelihood_test))
    df['AIC'].append(utility.calc_AIC(num_parameters, log_likelihood_test))

    return df

def box_plot_accuracy_time(df, y_string, hue_string, plot_GMLVQ = True):
    
    # load plot styles and extract
    general_styles= load_general_styles()
    
    # create grouped boxplot
    df_random = df.loc[df['init_type'] == 'random']
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(x = df_random['dimension'], 
                y = df_random[y_string],
                hue = df_random[hue_string],
                palette = general_styles['palette_box_plot'],
                showmeans=True,
                showfliers=True,
                ax=axs[0])
    
    # plot GMLVQ results on top
    if plot_GMLVQ:
        axs[0].legend([],[], frameon=False)
        df_GMLVQ = df.loc[df['init_type'] == 'GMLVQ']
        sns.boxplot(x = df_GMLVQ['dimension'],
                    y = df_GMLVQ[y_string],
                    hue = df_GMLVQ[hue_string],
                    palette = general_styles['palette_dark'],
                    fill=False, 
                    gap=.1,
                    showfliers=True,
                    ax=axs[0])
    
    sns.boxplot(x = df['dimension'], 
                y = df['time'],
                hue = df[hue_string],
                palette = general_styles['palette_box_plot'],
                showmeans=True,
                showfliers=True,
                ax=axs[1])
    
    return fig, axs

def box_plot_accuracy_time_v2(df, y_string, hue_string):
    
    # load plot styles and extract
    general_styles = load_general_styles()
    restricted_unrestricted_styles = load_restricted_unrestricted_styles()
    #sns.set_context(rc = {'patch.linewidth': 1.})

    # create grouped boxplot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    box_acc = sns.boxplot(
                x = df['dimension'], 
                y = df[y_string],
                hue = df[hue_string],
                hue_order=restricted_unrestricted_styles['hue_order'],
                palette = restricted_unrestricted_styles['palette'],
                legend=False,
                ax=axs[0])

    box_time = sns.boxplot(
                x = df['dimension'], 
                y = df['time'],
                hue = df[hue_string],
                hue_order=restricted_unrestricted_styles['hue_order'],
                palette = restricted_unrestricted_styles['palette'],
                legend=False,
                ax=axs[1])

    # define hatches
    hatches = ['xx', 'xx', 'xx', '', '', '', 'xx', 'xx', 'xx', '', '', '', '', '', '', '']

    # loop over the bars and set hatches
    for i, thisbar in enumerate(box_acc.patches):
        #thisbar.set_hatch(hatches[i])

        if i == 0:
            thisbar.set_label('a) spherical $K_{max}=5$')
        elif i == 3:
            thisbar.set_label('b) spherical $K_{max}=20$')
        elif i == 6:
            thisbar.set_label('c) full $K_{max}=5$')
        elif i == 10:
            thisbar.set_label('d) full $K_{max}=20$')
    
    return fig, axs

def bar_plot_accuracy_time(df, y_string, hue_string):
    
    # load plot styles and extract
    general_styles = load_general_styles()
    restricted_unrestricted_styles = load_restricted_unrestricted_styles()
    sns.set_context(rc = {'patch.linewidth': 1.})

    # create grouped boxplot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    bar_acc = sns.barplot(
                x = df['dimension'], 
                y = df[y_string],
                hue = df[hue_string],
                hue_order=restricted_unrestricted_styles['hue_order'],
                palette = restricted_unrestricted_styles['palette'],
                linewidth=1, 
                edgecolor="0",
                legend=False,
                ax=axs[0])

    bar_time = sns.barplot(
                x = df['dimension'], 
                y = df['time'],
                hue = df[hue_string],
                hue_order=restricted_unrestricted_styles['hue_order'],
                palette = restricted_unrestricted_styles['palette'],
                linewidth=1, 
                edgecolor="0",
                legend=False,
                ax=axs[1])

    # define hatches
    hatches = ['xx', 'xx', 'xx', '', '', '', 'xx', 'xx', 'xx', '', '', '', '', '', '', '']

    # loop over the bars and set hatches
    for i, thisbar in enumerate(bar_acc.patches):
        thisbar.set_hatch(hatches[i])

        if i == 0:
            thisbar.set_label('spherical res.')
        elif i == 3:
            thisbar.set_label('spherical unres.')
        elif i == 6:
            thisbar.set_label('full res.')
        elif i == 10:
            thisbar.set_label('full unres.')
    
    return fig, axs