import os
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread

def extract_number(file_name):
    # Extract numeric part from the file name
    numeric_part = ''.join(filter(str.isdigit, file_name))
    if numeric_part:
        return int(numeric_part)
    return 0

def plot_files(directory_path, n, k, img_row):
    # Get a list of files in the directory
    files = sorted([file for file in os.listdir(directory_path) if file.startswith('iter')])

    # Take the first n files
    files_to_plot = files[:n][::k]

    # Calculate the number of rows based on the total number of files
    num_rows = ((len(files_to_plot) + img_row ) // img_row) - 1

    # Create subplots with the dynamically calculated number of rows
    fig, axes = plt.subplots(num_rows, img_row, figsize=((5 * img_row)+1, (5 * num_rows)+1))

    # Iterate through files and plot them
    for i, file_name in enumerate(files_to_plot):
        row = i // img_row
        col = i % img_row
        if num_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        file_path = os.path.join(directory_path, file_name)

        # Read and plot the image
        image = imread(file_path)
        ax.imshow(image)
        ax.axis('off')

        # Extract the numeric part from the file name as the title
        numeric_part = extract_number(file_name)
        ax.set_title(f'{numeric_part}', fontsize=25, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save the figure in the same path
    figure_path = os.path.join(directory_path, 'plot_figure.png')
    plt.savefig(figure_path)

    # Show the plot (optional)
    # plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_script.py <directory_path> <n> <img_row>")
        sys.exit(1)

    directory_path = sys.argv[1]
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    img_row = int(sys.argv[4])

    plot_files(directory_path, n, k, img_row)
