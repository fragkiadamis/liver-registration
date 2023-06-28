import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import setup_parser, validate_paths, create_dir


def box_plots(data, output_dir, column, temp=None):
    output = create_dir(output_dir, "boxplots")

    # Calculate mean and median
    mean = np.mean(data)
    mean2 = np.mean(temp)
    median = np.median(data)
    median2 = np.median(temp)

    # Create box plots
    fig, ax = plt.subplots()
    if temp is not None:
        ax.boxplot([temp, data], vert=True, showmeans=True, meanline=True, labels=['B-Spline', 'LocalNet'])
    else:
        ax.boxplot(data, vert=False, showmeans=True, meanline=True, labels=['Data'])

    # Add mean and median markers
    ax.scatter([2, 1], [mean, mean2], color='red', marker='o', label='Mean')
    ax.scatter([2, 1], [median, median2], color='blue', marker='o', label='Median')

    # Set plot title and labels
    ax.set_title('LocalNet Box Plot')
    ax.set_xlabel('Value')

    # Show legend
    ax.legend()

    # Save the plot as a PNG image
    filename = f"box_plot_{column}.png"
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
            'Initial': 'initial_dice',
            'Unnamed: 2': 'initial_hd',
            '01_Affine_KS': 'affine_dice',
            'Unnamed: 4': 'affine_hd',
            '02_B-Spline_MI': 'bspline_dice',
            'Unnamed: 6': 'bspline_hd',
            '02_LocalNet (256)': 'localnet_256_dice',
            'Unnamed: 8': 'localnet_256_hd',
            '02_LocalNet (RS 512)': 'localnet_512_dice',
            'Unnamed: 10': 'localnet_512_hd'
        },
        inplace=True
    )
    for column in dataframe:
        if "patient" in column:
            continue

        box_plots(dataframe[column], output_dir, column, temp=dataframe['bspline_dice'])
        # plot_distribution(dataframe[column], output_dir, column)
        # plot_std(dataframe[column], output_dir, column)


if __name__ == "__main__":
    main()
