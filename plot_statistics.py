import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import setup_parser, validate_paths, create_dir


def box_plots(data, output_dir):
    output = create_dir(output_dir, "boxplots")

    # Calculate mean and median
    mean = np.mean(data[0])
    mean2 = np.mean(data[1])
    mean3 = np.mean(data[2])
    median = np.median(data[0])
    median2 = np.median(data[1])
    median3 = np.median(data[2])

    # Create box plots
    fig, ax = plt.subplots()
    ax.boxplot([data[0], data[1], data[2]], vert=True, showmeans=True, meanline=True, labels=['B-Spline', 'Expert', 'LocalNet'])
    ax.scatter([1, 2, 3], [mean, mean2, mean3], color='red', marker='o', label='Mean')
    ax.scatter([1, 2, 3], [median, median2, median3], color='blue', marker='o', label='Median')

    # Set plot title and labels
    ax.set_title('Results Box Plot')
    ax.set_xlabel('Value')

    # Show legend
    ax.legend()

    # Save the plot as a PNG image
    filename = f"box_plot.png"
    plt.savefig(f"{output}/{filename}", dpi=300, bbox_inches='tight')


def plot_distribution(data, output_dir, column):
    output = create_dir(output_dir, "distributions")

    # Plot histogram
    plt.hist(data, bins=30, edgecolor='black', alpha=0.75)

    # Set plot title and labels
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Save the plot as a PNG image
    filename = f"box_plot_{column}.png"
    plt.savefig(f"{output}/{filename}", dpi=300, bbox_inches='tight')


def plot_std(data, output_dir, column):
    output = create_dir(output_dir, "std")

    # Calculate standard deviation
    std = np.std(data)

    # Plot the data points
    plt.plot(data, 'o', markersize=3, label='Data')

    # Plot the standard deviation as error bars
    plt.errorbar(range(len(data)), data, yerr=std, linestyle='None', ecolor='red', capsize=3,
                 label='Standard Deviation')

    # Set plot title and labels
    plt.title('Data with Standard Deviation')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Add a legend
    plt.legend()

    # Save the plot as a PNG image
    filename = f"box_plot_{column}.png"
    plt.savefig(f"{output}/{filename}", dpi=300, bbox_inches='tight')


def main():
    dir_name = os.path.dirname(__file__)
    args = setup_parser(f"{dir_name}/config/statistics_parser.json")
    input_file = os.path.join(dir_name, args.i)
    output_dir = os.path.join(dir_name, args.o)

    # Validate paths, create structure and open the dataframe.
    validate_paths(input_file, output_dir)
    create_dir(dir_name, output_dir)

    dataframe = pd.read_excel(input_file)[2:70]
    dataframe.rename(
        columns={
            'Unnamed: 0': 'patients',
            'Initial': 'liver_initial_dice',
            'Unnamed: 2': 'liver_initial_hd',
            'Unnamed: 3': 'tumor_initial_dice',
            'Unnamed: 4': 'tumor_initial_hd',
            'Unnamed: 5': 'tumor_bb_initial_dice',
            'Unnamed: 6': 'tumor_bb_initial_hd',
            'Affine': 'liver_affine_dice',
            'Unnamed: 8': 'liver_affine_hd',
            'Unnamed: 9': 'tumor_affine_dice',
            'Unnamed: 10': 'tumor_affine_hd',
            'Unnamed: 11': 'tumor_bb_affine_dice',
            'Unnamed: 12': 'tumor_bb_affine_hd',
            'B-Spline': 'liver_bspline_dice',
            'Unnamed: 14': 'liver_bspline_hd',
            'Unnamed: 15': 'tumor_bspline_dice',
            'Unnamed: 16': 'tumor_bspline_hd',
            'Unnamed: 17': 'tumor_bb_bspline_dice',
            'Unnamed: 18': 'tumor_bb_bspline_hd',
            'LocalNet': 'liver_localnet_dice',
            'Unnamed: 20': 'liver_localnet_hd',
            'Unnamed: 21': 'tumor_localnet_dice',
            'Unnamed: 22': 'tumor_localnet_hd',
            'Unnamed: 23': 'tumor_bb_localnet_dice',
            'Unnamed: 24': 'tumor_bb_localnet_hd',
            "Expert": 'liver_expert_dice',
            'Unnamed: 26': 'liver_expert_hd',
            'Unnamed: 27': 'tumor_expert_dice',
            'Unnamed: 28': 'tumor_expert_hd',
            'Unnamed: 29': 'tumor_bb_expert_dice',
            'Unnamed: 30': 'tumor_bb_expert_hd'
        },
        inplace=True
    )
    box_plots([dataframe['tumor_bspline_dice'], dataframe['tumor_expert_dice'], dataframe['tumor_localnet_dice']], output_dir)
    for column in dataframe:
        if "patient" in column:
            continue
        print(column)

        # plot_distribution(dataframe[column], output_dir, column)
        # plot_std(dataframe[column], output_dir, column)


if __name__ == "__main__":
    main()
