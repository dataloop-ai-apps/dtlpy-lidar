import os
import pickle
import pandas as pd


def convert_pkl_to_csv(input_folder, output_folder):
    # List all .pkl files in the input folder
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]

    if len(pkl_files) == 0:
        print("No .pkl files found in the folder.")
        return

    # Process each .pkl file
    for pkl_file in pkl_files:
        pkl_file_path = os.path.join(input_folder, pkl_file)
        csv_file_path = os.path.join(output_folder, pkl_file.replace('.pkl', '.csv'))

        try:
            # Load the .pkl file
            with open(pkl_file_path, 'rb') as file:
                data = pickle.load(file)

            # Convert data to DataFrame if needed
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)

            # Save to .csv file
            df.to_csv(csv_file_path, index=False)
            print(f"Successfully converted '{pkl_file_path}' to '{csv_file_path}'")

        except Exception as e:
            print(f"Failed to process {pkl_file_path}: {e}")


def test_convert_pkl_to_csv():
    input_folder = r"pandaset/001/annotations/cuboids"
    output_folder = r"pandaset/001/annotations/cuboids"
    convert_pkl_to_csv(input_folder=input_folder, output_folder=output_folder)


if __name__ == "__main__":
    test_convert_pkl_to_csv()
