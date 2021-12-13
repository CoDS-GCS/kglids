import io
import pandas as pd
from subprocess import check_output, CalledProcessError, STDOUT
import shlex
import os
from tqdm import tqdm
import kaggle  # make sure kaggle package is downloaded

# to keep first 10 rows of dataset:
# for i in *.csv; do echo "$(head -10 $i)" > $i; done


def syscall(command):
    """
    params:
        command: string, ex. `"ls -l"`
    returns: output, success
    """
    command = shlex.split(command)
    try:
        output = check_output(command, stderr=STDOUT).decode()
        success = True
    except CalledProcessError as e:
        output = e.output.decode()
        success = False
    return output, success


def main():

    datasets = pd.read_csv('data/kaggle_datasets.csv')

    for _, row in tqdm(datasets.iterrows(), total=len(datasets)):

        dataset_name = row['ref']
        dataset_path = 'data/kaggle/' + dataset_name.replace('/', '.')

        # Downlaod the dataset (if size is < 500MB)
        size = row['size']
        if 'GB' in size or ('MB' in size and int(size[:size.index('MB')]) > 500):
            print('Dataset too big:', dataset_name, size)
        else:
            download_cmd = f'kaggle datasets download {dataset_name} --path {dataset_path}/data/ --unzip -q'
            # download the dataset file
            output, succes = syscall(download_cmd)
            if not succes:
                print('Error while downloading dataset:', dataset_name)
                print(output)

        # List the notebooks
        notebooks = []
        for i in range(1, 7):
            list_cmd = f'kaggle kernels list --dataset {dataset_name} --page {i} --sort-by voteCount --language python -v'
            output, succes = syscall(list_cmd)
            if output == 'No kernels found':
                break
            df = pd.read_csv(io.StringIO(output))
            notebooks.append(df)

        notebooks = pd.concat(notebooks, ignore_index=True).drop_duplicates('ref')

        # download 100 kernels
        for _, nb_row in tqdm(notebooks.iterrows(), total=len(notebooks)):
            nb_name = nb_row['ref']
            nb_download_cmd = f'kaggle kernels pull {nb_name} --path {dataset_path}/notebooks/'
            output, succes = syscall(nb_download_cmd)
            if not succes:
                print('Error while downloading kernel:', nb_name, '; Dataset:', dataset_name)
                continue
            if len(os.listdir(f'{dataset_path}/notebooks')) >= 100:
                break

if __name__ == '__main__':
    main()
