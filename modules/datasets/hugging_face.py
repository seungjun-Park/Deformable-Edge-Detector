import glob
import os
import zipfile
import shutil

from huggingface_hub import hf_hub_download
# from waifuc.source import LocalSource

characters = ['wakamo']
dataset_path = '../../datasets/anime'

for name in characters:
    # download raw archive file
    zip_file = hf_hub_download(
        repo_id=f'CyberHarem/{name}_bluearchive',
        repo_type='dataset',
        filename='dataset-raw.zip',
    )

    # extract files to your directory
    os.makedirs(f'{dataset_path}/{name}', exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(f'{dataset_path}/{name}')

    os.makedirs(f'{dataset_path}/train/{name}/edges', exist_ok=True)
    os.makedirs(f'{dataset_path}/train/{name}/images', exist_ok=True)
    os.makedirs(f'{dataset_path}/train/{name}/tags', exist_ok=True)

    os.makedirs(f'{dataset_path}/val/{name}/edges', exist_ok=True)
    os.makedirs(f'{dataset_path}/val/{name}/images', exist_ok=True)
    os.makedirs(f'{dataset_path}/val/{name}/tags', exist_ok=True)

    images = glob.glob(f'{dataset_path}/{name}/*.*')
    tags = glob.glob(f'{dataset_path}/{name}/.*.*')

    for i, (image, tag) in enumerate(zip(images, tags)):
        if i < 5:
            split = 'train'
        else:
            split = 'val'

        tag, tag_form = tag.rsplit('.', 1)
        tag_src, tag = tag.split('\\', 1)

        image, image_form = image.rsplit('.', 1)
        image_src, image = image.split('\\', 1)

        shutil.move(f'{image_src}/{image}.{image_form}', f'{dataset_path}/{split}/{name}/images/{i}.{image_form}')
        shutil.move(f'{tag_src}/{tag}.{tag_form}', f'{dataset_path}/{split}/{name}/tags/{i}.{tag_form}')

    os.rmdir(f'{dataset_path}/{name}')