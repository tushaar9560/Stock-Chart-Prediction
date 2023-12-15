import os

class Config:
    base_path = 'artifacts'
    images_path = os.path.sep.join([base_path, 'images'])
    annots_path = os.path.sep.join([base_path, 'annotations.csv'])

    base_output = 'output'
    model_path = os.path.sep.join([base_output, 'detectorBB.h5'])
    plot_path = os.path.sep.join([base_output, 'plot.png'])
    test_filenames = os.path.sep.join([base_output, 'test_images.txt'])

    init_lr = 1e-4
    epochs = 25
    batch_size = 3