import os
import numpy as np
import h5py
from skimage.io import imread
from datetime import datetime
from tqdm.auto import tqdm
import cv2
from typing import Callable, Optional

from .utils import preprocess_volume, preprocess_mask, save_data

class Dataset():
    """
    Dataset of brain MRIs for Cytotoxic oedema (Stroke) segmentation.

    Attributes:
        IMG_SHAPE : tuple
            Shape of the images: (H, W).
        VOLUMES : list
            Names of the volumes.

    Methods:
        make_dataset(raw_data_dir='./raw_data', data_dir='./data')
            Creates a virtual HDF5 dataset with preprocessed images and metadata.
    """

    IMG_SHAPE = (256, 256)
    VOLUMES = ["DWI", "ADC"]

    def __init__(self, 
                 path="~/Documents/bolnica/Stroke-nn_working/data/brainMRI.h5",
                 dataset = None,
                 ds_type='Train', 
                 volume=None, 
                 seed=42):
        """Initializes the brain MRI dataset.
            
        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is './data/brainMRI.h5'.
            train : bool, optional
                Slice of the dataset to select.
                Default is True, selects 80% of the patients.
            volume : str, optional.
                Volume images to return.
                Default is None, returns all the volumes.
            seed : int, optional.
                Seed for the random number generator to split
                the data into train and test sets.

        Returns: instance of the brain MRI dataset
        """

        if not os.path.exists(path) and dataset is None:
            raise RuntimeError("Dataset not found at '{}'.\nUse Dataset.make_dataset() to create it.".format(path))

        assert volume in [None,] + Dataset.VOLUMES, 'volume can only be None or one of {}'.format(Dataset.VOLUMES)

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = h5py.File(path, 'r')
        self.volume = volume
        
        self.index_vector = np.arange(len(self.dataset['slices']))
        patients = np.unique(self.patients)
        #rng = np.random.default_rng(seed=seed)
        #rng.shuffle(patients)
        np.random.seed(seed)
        np.random.shuffle(patients)
        
        train_size = 0.7
        valid_size = 0.3 # 1 - 0.7 = 0.3
        
        
        #start_idx = 30
        #end_idx = 100
        
        #test_ds, valid_ds, train_ds = np.split(patients, [start_idx, end_idx])
        
        #train_ds = np.concatenate((test_ds, train_ds))
        
        train_end = int(train_size * len(patients))
        valid_end = int((train_size + valid_size) * len(patients))

        train_ds, valid_ds, test_ds = np.split(patients, [train_end, valid_end])
        print(f"Total patients: {len(patients)}")
        print(f"Train patients: {len(train_ds)}")
        print(f"Validation patients: {len(valid_ds)}")
        print(f"Test patients: {len(test_ds)}")
       
        if ds_type == 'Train':
            data_patients = train_ds
            print(f'Selected {data_patients.size} for training')
        elif ds_type == 'Validation':
            data_patients = valid_ds
            print(f'Selected {data_patients.size} for validation')
        elif ds_type == 'Test':
            data_patients = test_ds
            print(f'Selected {data_patients.size} for test')
        
        print(f'Train/Validation/Test %: {int(train_ds.size*100/patients.size)}/{int(valid_ds.size*100/patients.size)}/{int(test_ds.size*100/patients.size)}')
        
        bool_vector = np.zeros_like(self.index_vector)
        for patient in data_patients:
            bool_vector = bool_vector + np.array(self.patients == patient)
        self.index_vector = np.where(bool_vector)[0]

    def __getitem__(self, index):
        index = self.index_vector[index]

        img = self.dataset['images'][index].astype(np.float32)
        mask = self.dataset['masks'][index]
        patient = self.dataset['patients'][index]
        slice = self.dataset['slices'][index]

        if self.volume is not None:
            img = img[self.VOLUMES.index(self.volume)]
            img = img[np.newaxis, ...]

        return img, mask, (patient, slice)

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        return self.dataset["images"][self.index_vector]

    @property
    def masks(self):
        return self.dataset["masks"][self.index_vector]

    @property
    def patients(self):
        return self.dataset["patients"][self.index_vector]

    @property
    def slices(self):
        return self.dataset["slices"][self.index_vector]

    @staticmethod
    def make_dataset(raw_data_dir='./raw_images', data_dir='./data'):
        """Creates a virtual HDF5 dataset with preprocessed images and metadata.

        Args:
            raw_data_dir : str, optional
                Path to the raw data directory.
            data_dir : str, optional
                Path to the processed data directory.
        """

        # check data directories
        if not os.path.exists(raw_data_dir):
            print('{} does not exist!'.format(
                raw_data_dir
            ))
            raise OSError
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        h5_dir = os.path.join(data_dir, 'hdf5')
        if not os.path.exists(h5_dir):
            os.mkdir(h5_dir)
            
        patient_dirs = [
            d
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(
                os.path.join(raw_data_dir, d)
            )
        ]

        pbar = tqdm(total=len(patient_dirs), desc='Retrieving data', leave=False)

        n_samples = 0
        mask_count = 0        
        for patient_dir in patient_dirs:
            
            # retrieve patient images and masks
            pbar.set_description(f'{patient_dir} [Retrieving data]')
            dir_path = os.path.join(raw_data_dir, patient_dir)
            img_names = [
                x
                for x in os.listdir(dir_path)
                if 'mask' not in x
            ]
            img_names.sort(
                key=lambda x: int(x.split(".")[0].split("_")[2]) #number in filename
            )
            n_slices = len(img_names)
            n_samples += n_slices
            
            images = np.empty((n_slices, len(Dataset.VOLUMES), *Dataset.IMG_SHAPE), dtype=np.float32)
            
            masks = np.empty((n_slices, *Dataset.IMG_SHAPE), dtype=np.uint8)
            for i, name in enumerate(img_names):
                img_path = os.path.join(dir_path, name)
                prefix, ext = os.path.splitext(img_path)
                mask_path = prefix + '_mask' + ext
                        
                img_u16 = imread(img_path)
                    
                img = []
                img.append(cv2.normalize(img_u16[0], None, 0.0, 1.0, cv2.NORM_MINMAX, dtype = cv2.CV_32F))
                img.append(cv2.normalize(img_u16[1], None, 0.0, 1.0, cv2.NORM_MINMAX, dtype = cv2.CV_32F))
                
                #logging
                save_data(name, img[0].dtype, img[1].dtype, 
                          np.min(img[0]), np.max(img[0]), np.mean(img[0]), 
                          np.min(img[1]), np.max(img[1]), np.mean(img[1]))
                
                images[i] = img
                masks[i] = imread(mask_path)
                
                if (masks[i].sum() > 1):
                    mask_count += 1
            
            # preprocess images and metadata
            pbar.set_description(f'{patient_dir} [Preprocessing data]')
            
            #images = preprocess_volume(images)
            
            masks = preprocess_mask(masks)
            
            patient = np.array((patient_dir,)*n_slices)
            slices = np.array([
                int(x.split('.')[0].split("_")[2]) #number in filename
                for x in img_names
            ], dtype=np.uint8)

            # create patient dataset
            pbar.set_description(f'{patient_dir} [Saving data]')
            h5_file_path = os.path.join(h5_dir, patient_dir + '.h5')
            with h5py.File(h5_file_path, 'w') as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['info'] = h5py.version.info             
                masks_data = np.moveaxis(masks[..., np.newaxis], 3, 1)             
                h5_file.create_dataset("images", data=images)             
                h5_file.create_dataset("masks", data=masks_data)        
                h5_file.create_dataset("patients", data=patient.astype(h5py.string_dtype(encoding='utf-8')))
                h5_file.create_dataset("slices", data=slices)

            
            pbar.update(1)
        print(f"Total masks {mask_count} of {n_samples} files")

        # create virtual layouts
        pbar.set_description('Creating virtual dataset')
        layouts = {
            "images": h5py.VirtualLayout(
                shape=(n_samples, len(Dataset.VOLUMES), *Dataset.IMG_SHAPE), 
                dtype=np.float16
                ),
            "masks": h5py.VirtualLayout(
                shape=(n_samples, 1, *Dataset.IMG_SHAPE), 
                dtype=np.uint8
                ),
            "patients": h5py.VirtualLayout(
                shape=(n_samples,), 
                dtype=h5py.string_dtype(encoding='utf-8')
                ),
            "slices": h5py.VirtualLayout(
                shape=(n_samples,), 
                dtype=np.uint8
                )
        }
        
        # fill the virtual layouts
        i = 0
        for filename in os.listdir(h5_dir):
            file_path = os.path.join(h5_dir, filename)
            with h5py.File(file_path, "r") as h5_file:
                n_slices = h5_file['slices'].shape[0]
                for k in h5_file.keys():
                    layouts[k][i:i+n_slices] = h5py.VirtualSource(h5_file[k])
                i += n_slices
        
        # create virtual dataset
        vds_path = os.path.join(data_dir, 'brainMRI.h5')
        with h5py.File(vds_path, "w") as h5_file:
            h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5_file.attrs['h5py_info'] = h5py.version.info
            h5_file.attrs['dataset'] = 'Stroke Brain MRI'
            for name, layout in layouts.items():
                h5_file.create_virtual_dataset(name, layout)

        pbar.close()
