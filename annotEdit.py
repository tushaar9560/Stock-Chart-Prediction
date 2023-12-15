import os
import re
import pandas as pd
# To convert provided label to single CSV file

def edit_annotations(annotation_dir, output_file):
    for filename in os.listdir(annotation_dir):
        with open(os.path.join(annotation_dir, filename), 'r') as file:
            content = file.read()
            new_content = re.sub(r"\b1\b", os.path.splitext(filename)[0], content)
        with open(os.path.join(annotation_dir, filename), 'w') as file:
            file.write(new_content)

    merge_annotations(annotation_dir, output_file)
    sort_annotations(output_file)

    with open(output_file,'r') as file:
        file_content = file.read()
    file_content = file_content.replace(' ', ',')
    with open(output_file, 'w') as file:
        file.write(file_content)

def merge_annotations(annotation_dir, output_file):
    with open(output_file, 'w') as output_file:
        for filename in os.listdir(annotation_dir):
            with open(os.path.join(annotation_dir, filename), 'r') as input_file:
                output_file.write(input_file.read())
                output_file.write('\n')

def sort_annotations(annotations_file):
    with open(annotations_file, 'r') as file:
        lines = file.readlines()

    # Remove empty lines
    non_empty_lines = [line for line in lines if line.strip() != '']

    # Sort the lines based on the first number (if present)
    sorted_lines = sorted(non_empty_lines, key=lambda line: float(line.split()[0]) if line.split() else 0)

    # Write back the sorted and non-empty lines
    with open(annotations_file, 'w') as file:
        file.writelines(sorted_lines)

def text_to_csv(input_file, output_dir):
    input_filename = os.path.splitext(os.path.basename(input_file))[0]

    csv_file_path = os.path.join(output_dir, input_filename + '.csv')

    df = pd.read_csv(input_file, delim_whitespace=True, header=None)
    df.to_csv(csv_file_path, index=False, header=False)
    os.remove(input_file)
    
if __name__ == "__main__":
    image_dir = "Images/"
    annotation_dir = "labels/"
    output_dir = "artifacts/"
    annotations_file = 'annotations.txt'

    edit_annotations(annotation_dir, annotations_file)
    text_to_csv(annotations_file, output_dir)