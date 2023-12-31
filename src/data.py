import numpy as np
from typing import List, Tuple, Dict
from ase import Atoms
from ase import Atoms
from ase.io import write
from pathlib import Path
from ase.io import read
import torch
from sklearn.preprocessing import StandardScaler
from ase.io import read, write

from scipy import stats

root_dir = Path(__file__).resolve().parent.parent


def get_mda_paper_structures(
    struct_type: str, run: int = 1, all_runs: bool = True
) -> List[Atoms]:
    """Get the structures from the MDA paper.

    Parameters
    ----------
    struct_type : str
        The type of structure to get. Can be "ice_Ih", "initial_shears", "final_shears", "mda".
    run : int, optional
        The run to get the shearing structures from; by default 1.
        If all_runs is False, then this is also the run from which the final MDA structure is taken.
        The paper published 5 shearing trajectories of Ice Ih -> MDA; so run can be 1, 2, 3, 4, or 5.
    all_runs : bool, optional
        Whether or not to get the final MDA structures from all 5 run.
        By default True as we will use all 5 MDA structures for training/testing.

    Returns
    -------
    List[Atoms]
        A list of the structures.
    """

    # Define a dictionary to map structure types to their respective file patterns
    file_patterns = {
        "initial_shears": range(
            1, 11
        ),  # the first 10 structures of the shear trajectory
        "final_shears": range(
            11, 101
        ),  # the remaining 90 structures of the shear trajectory
        "mda": [run] if not all_runs else range(1, 6),  # the final MDA structure
    }

    if struct_type in file_patterns:
        struct_list = []
        for file_id in file_patterns[struct_type]:
            if struct_type == "mda":
                file_name = (
                    root_dir / f"data/mda/run{file_id}/final_mda.extxyz"
                )
            else:
                file_name = (
                    root_dir / f"data/mda/run{run}/mda_conf{file_id}.extxyz"
                )
            struct_list.append(read(file_name))
        return struct_list
    else:
        struct_list = []
        file_name = (
            root_dir / f"data/{struct_type}/{struct_type}_conf1.extxyz"
        )
        struct_list.append(read(file_name))
        return struct_list


def write_to_extxyz(
    file_path: str, struct: Atoms, descriptor: np.ndarray, **kwargs
) -> None:
    """Write the structure to an extended xyz file along with the descriptors for each atom.
    Additional keyword arguments can be passed to the function to add more information to the file.
    E.g: write_to_extxyz(file_path, struct, descriptor, struct_type="hda", pressure=1, temp=190)


    Parameters
    ----------
    file_path : str
        The path to write the file to.
    struct : Atoms
        The structure.
    descriptor : np.ndarray
        The descriptor for each atom.
    **kwargs
        Additional keyword arguments to add to the file.
    """
    struct.arrays["steinhardt_descriptor"] = descriptor

    for key, value in kwargs.items():
        struct.info[key] = value

    write(
        file_path,
        struct,
        format="extxyz",
    )

def label_to_one_hot(labels: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Convert the labels to one-hot vectors.
    Structure types are mapped to integers, and then converted to one-hot vectors.
    E.g. "hda": 0, "lda": 1, "liquid": 2, "mda": 3

    Parameters
    ----------
    labels : np.ndarray
        Array of labels to convert.

    Returns
    -------
    np.ndarray
        Array of one-hot vectors; one for each label.
    label_mapping : dict
        Mapping from structure type to integer.
    """
    # find the unique labels and map them to integers
    # returns them in alphabetical order: hda, lda, liquid, mda
    unique_labels = np.unique(labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    # initialise array of zeros
    one_hot_vectors = np.zeros((len(labels), len(label_mapping)))
    for i, label in enumerate(labels):
        one_hot_vectors[
            i, label_mapping[label]
        ] = 1  # set the correct index to 1, e.g. [0, 1, 0, 0] if label is "lda"
    return one_hot_vectors, label_mapping


def one_hot_to_label(one_hot_vectors: np.ndarray, label_mapping: dict) -> np.ndarray:
    """Convert the one-hot vectors to labels.

    Parameters
    ----------
    one_hot_vectors : np.ndarray
        Array of one-hot vectors to convert.
    label_mapping : dict
        Mapping from structure type to integer.

    Returns
    -------
    np.ndarray
        Array of labels; one for each one-hot vector.
    """
    label_indices = np.array(np.argmax(one_hot_vectors, axis=1))
    labels = [label_mapping[index] for index in label_indices]
    return np.array(labels)


def shuffle_files_from_dir(data_type: str, struct_type: str) -> List[Path]:
    """Shuffle the files in a directory.

    Parameters
    ----------
    data_type : str
        The type of data to get. Can be training or testing.
    struct_type : str
        The type of structure to get. Can be "lda", "liquid", or "hda".

    Returns
    -------
    List[Path]
        A list of the shuffled file paths.
    """
    # get all the files from the directory
    path_to_data = root_dir / f"data/{data_type}/{struct_type}"
    all_files = sorted(path_to_data.iterdir())

    # shuffle the files
    np.random.seed(42)
    np.random.shuffle(all_files)
    return all_files


def get_train_val_test_structures(
    struct_type: str, num_files: int = None
) -> Tuple[List[Atoms], List[Atoms], List[Atoms]]:
    """Get the training, validation, and testing structures for the given structure type.

    Parameters
    ----------
    struct_type : str
        The type of structure to get data for. Can be "lda", "liquid", or "hda".
    num_files : int
        The number of files (i.e. structures) to get. Default is None - i.e. all files.
        Specifying a smaller number of files will reduce the memory and requirement and speed up training.

    Returns
    -------
    Tuple[List[Atoms], List[Atoms], List[Atoms]]
        The training, validation, and testing structures.
    """
    # shuffle all the files
    # needed as we're only going to take a subset of the files
    # need to make sure we're not accidentally taking trajectories/similar structures
    all_train_files = shuffle_files_from_dir("training", struct_type)
    all_test_files = shuffle_files_from_dir("testing", struct_type)

    if num_files is not None:
        all_train_files = all_train_files[:num_files]
        all_test_files = all_test_files[:num_files]

    # take 10 structures for validation
    val_files = all_train_files[-10:]
    # take the rest for training
    train_files = [file for file in all_train_files if file not in val_files]

    train_structs = [read(file) for file in train_files]
    val_structs = [read(file) for file in val_files]
    test_structs = [read(file) for file in all_test_files]

    # remove the anomalous HDA structures
    # there are some highly crystalline structures among the HDA structures
    # these are anomalous and should be removed
    if struct_type == "hda":
        train_densities = [calc_water_struct_density(s) for s in train_structs]
        val_densities = [calc_water_struct_density(s) for s in val_structs]
        test_densities = [calc_water_struct_density(s) for s in test_structs]

        def filter_structures(structs, densities):
            return [s for s, density in zip(structs, densities) if density > 1.15]

        filtered_train_structs = filter_structures(train_structs, train_densities)
        filtered_val_structs = filter_structures(val_structs, val_densities)
        filtered_test_structs = filter_structures(test_structs, test_densities)
    else:
        filtered_train_structs = train_structs
        filtered_val_structs = val_structs
        filtered_test_structs = test_structs

    return filtered_train_structs, filtered_val_structs, filtered_test_structs


def train_val_test_split(
    struct_types: List[str] = ["lda", "hda"], num_files: int = None, mda=True
) -> Tuple[List[Atoms], List[Atoms], List[Atoms]]:
    """Get the training, validation, and testing data for the water configurations.
    The training set will consist of num_files of each structure type.

    Parameters
    ----------
    struct_types : List[str], optional
        The structure types to get data for, by default ["lda", "hda"].
        Can change, for example, to include high-T liquid structures in training.
        Structure types can be "lda", "liquid", or "hda".
    num_files : int
        Number of files to use for training. Default is None - i.e. all files.
    mda : bool, optional
        Whether to include the MDA structures in training, by default True.

    Returns
    -------
    Tuple[List[Atoms], List[Atoms], List[Atoms]]
        The training, validation, and testing data.
    """

    train_structs = []
    val_structs = []
    test_structs = []

    # Get the training, validation, and testing structures for each structure type
    for s in struct_types:
        train, val, test = get_train_val_test_structures(s, num_files)
        train_structs += train
        val_structs += val
        test_structs += test

    # Get the MDA structures
    if mda:
        # get the final MDA structure from all 5 runs
        mda_structs = get_mda_paper_structures(
            struct_type="mda", all_runs=True
        )
        # take the first 3 for training, the 4th for validation, and the 5th for testing

        train_structs += mda_structs[:3]
        val_structs += [mda_structs[3]]
        test_structs += [mda_structs[4]]

    return train_structs, val_structs, test_structs


def get_descriptor_and_labels(
    structures: List[Atoms],
    num_samples_per_type: int = 8_000,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, np.ndarray]:
    """Get the descriptor and labels for the given structures.

    Parameters
    ----------
    structures : List[Atoms]
        The structures to get the descriptors and labels for.
    num_samples_per_type : int, optional
        The number of samples to take from each structure, by default 8_000.
        This corresponds to the number of atoms of each structure type 
        that will be used for training/validation/testing.

    Returns
    -------
    Tuple[torch.FloatTensor, torch.FloatTensor, np.ndarray]
        The descriptor, labels, and label mapping.
        The label mapping is a dictionary that maps the structure type to an integer.
        It tells you what order the one-hot vectors are in.
    """

    struct_labels = []
    descriptors = []
    for atoms in structures:
        descriptors.append(atoms.arrays["steinhardt_descriptor"])
        # add the structure type as a label for each atom
        struct_labels.append([atoms.info["struct_type"]] * len(atoms))

    all_descriptors = np.vstack(descriptors)  # shape (numb_atoms, descriptor_length)
    all_labels = np.concatenate(struct_labels)  # shape (numb_atoms,)

    assert len(all_descriptors) == len(
        all_labels
    ), "The number of descriptors and labels must be the same - one per atom."

    unique_labels = np.unique(all_labels)

    # Initialize a dictionary to store data for each structure type
    descriptor_dict = {label: [] for label in unique_labels}

    # Split data by structure type
    for descriptor, label in zip(all_descriptors, all_labels):
        descriptor_dict[label].append(descriptor)

    # Shuffle and select a subset of points from each structure type
    descriptor_samples = []
    label_samples = []
    for label, data_list in descriptor_dict.items():
        if num_samples_per_type == None:
            num_samples_per_type = len(data_list)

        indices = np.arange(len(data_list))
        permutation = np.random.RandomState(seed=42).permutation(indices)
        if num_samples_per_type > len(data_list):
            print(
                f"There are fewer than {num_samples_per_type} samples for {label}. Taking the whole dataset."
            )
        shuffled_data = np.array(data_list)[permutation]
        shuffled_labels = np.array([label] * len(data_list))[permutation]

        # select a subset
        descriptor_samples.extend(shuffled_data[:num_samples_per_type])
        label_samples.extend(shuffled_labels[:num_samples_per_type])

    one_hot_vectors, label_mapping = label_to_one_hot(label_samples)

    x = torch.FloatTensor(np.array(descriptor_samples))
    y = torch.FloatTensor(one_hot_vectors)

    return x, y, label_mapping


def calc_water_struct_density(struct: Atoms) -> float:
    """Calculate the density of the water structure.
    This function assumes that:
        - the structure is an ase.Atoms object,
        - the unit cell parameters are in Angstrom,
        - the atoms object contains only oxygen atoms.

    Parameters
    ----------
    struct : Atoms
        Structure to calculate the density for.

    Returns
    -------
    float
        The density of the structure in g/cm^3.
    """
    numb_oxygens = len(struct)
    numb_hydrogens = 2 * numb_oxygens

    mass_oxygen = 15.999 * 1.66053906660e-24  # in g
    mass_hydrogen = 1.008 * 1.66053906660e-24  # in g

    total_mass = numb_oxygens * mass_oxygen + numb_hydrogens * mass_hydrogen  # in g

    volume = struct.get_volume()  # in Angstrom^3

    density = total_mass / (volume * 1e-24)  # in g/cm^3

    return density


def process_water_structures(
    data_type: int,
) -> Tuple[Dict[str, np.ndarray]]:
    """Process the water structures for plotting KDE's.
    This function returns a dictionary which is split by structure types.

    Parameters
    ----------
    data_type : int
        The type of data to get. Can be training or testing.
        The plots will look very similar for both training and testing data.

    Returns
    -------
    Tuple[Dict[str, np.ndarray]]
        The descriptors per atom for each structure type.
    """
    struct_types = ["hda", "lda"] # can add "liquid" here if you want to include liquid structures

    descriptor_params = {}
    for type in struct_types:
        path_to_data = root_dir / f"data/{data_type}/{type}"
        file_names = path_to_data.iterdir()

        structs = [read(f) for f in file_names]

        if type == "hda":
            # remove the anomalous HDA structures
            # there are some highly crystalline structures among the HDA structures
            # these are anomalous and should be removed
            def filter_structures(structs, densities):
                return [s for s, density in zip(structs, densities) if density > 1.15]

            densities = [calc_water_struct_density(s) for s in structs]

            main_hda_structs = filter_structures(structs, densities)
            descriptor_params[type] = np.vstack(
                [s.arrays["steinhardt_descriptor"] for s in main_hda_structs]
            )
        else:
            descriptor_params[type] = np.vstack(
                [s.arrays["steinhardt_descriptor"] for s in structs]
            )
        
    return descriptor_params

def process_mda_structures(
    all_runs: bool = True,
    run: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Process the mda dataset.
    This function returns a dictionary which is split by structure types.
    The structure types aim to follow the protocol used in the MDA paper:
        - Ice Ih
        - Initial shears (first 10 structures)
        - Final shears (last 90 structures)
        - MDA

    Parameters
    ----------
    all_runs : bool, optional
        Whether to include the MDA structures from all 5 runs, by default True
    run : int, optional
        The run to get the shearing data from; by default 1.
        If all_runs is False, then this is also the run from which the MDA structure is taken.

    Returns
    -------
    Tuple[Dict[str, np.ndarray]]
        The descriptors per atom each structure type.
    """
    descriptor_params = {}

    mda_struct_types = [
        "ice_Ih",
        "initial_shears",
        "final_shears",
        "mda",
    ]
    
    for type in mda_struct_types:
        structs = get_mda_paper_structures(struct_type=type, run=run, all_runs=all_runs)
        descriptor_params[type] = np.vstack(
            [s.arrays["steinhardt_descriptor"] for s in structs]
        )

    return descriptor_params


def random_sample_from_dict(data_dict, property_dict, sample_size=5000, seed=42):
    sampled_descriptors = {}
    sampled_property = {}

    for s in data_dict.keys():
        random_order = np.random.RandomState(seed=seed).permutation(
            data_dict[s].shape[0]
        )
        sample = random_order[:sample_size]
        sampled_descriptors[s] = data_dict[s][sample]
        sampled_property[s] = property_dict[s][sample]

    all_steinhardt = np.vstack(list(sampled_descriptors.values()))
    all_densities = np.hstack(list(sampled_property.values()))

    return all_steinhardt, all_densities, sampled_descriptors, sampled_property


def get_kde(
    data: np.ndarray, x_values: np.ndarray, bw_method=None, normalise: bool = True
) -> np.ndarray:
    """Get the kernel density estimate for the given data.
    The density estimates are taken at the given x_values.

    Parameters
    ----------
    data : np.ndarray
        The data to get the density estimate for.
    x_values : np.ndarray
        The values to get the density estimates at.
    bw_method : _type_, optional
        The bandwidth to use for the kernel density estimate; by default None, which uses the default value.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    normalise : bool, optional
        Whether to normalise the density estimates, by default True.
        Use this when plotting so that all the kde's have the same height of 1.

    Returns
    -------
    np.ndarray
        The density estimates at the given x_values.
    """
    kde = stats.gaussian_kde(data, bw_method=bw_method)
    density_estimates = kde.evaluate(x_values)

    if normalise:
        density_estimates /= np.max(density_estimates)

    return density_estimates

def predict_test_set_classes(
    test_structs: List[Atoms], scaler: StandardScaler, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the structure classes for the given test structures.

    Parameters
    ----------
    test_structs : List[Atoms]
        The test structures to predict the classes for.
    scaler : StandardScaler
        The scaler used to standardise the descriptors.
    model : 
        The trained model to use for prediction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The predicted classes, the true classes, and the confidence for each prediction.
    """
    pred_classes = []
    test_labels = []
    pred_confidences = []

    with torch.no_grad():
        for s in test_structs:
            desc = s.arrays["steinhardt_descriptor"]
            
            test_x = torch.FloatTensor(scaler.transform(desc))
            # get the predictions
            # apply softmax to get the % confidence for each class
            pred_y = torch.nn.Softmax(dim=-1)(model(test_x))

            # get the class with the highest confidence
            pred_class = pred_y.argmax(dim=-1).numpy()
            # get the % confidence for the predicted class
            confidences = pred_y[np.arange(len(pred_y)), pred_class].numpy()

            struct_type = s.info["struct_type"]
            test_labels += [struct_type] * len(s)

            pred_classes.extend(pred_class)
            pred_confidences.extend(confidences)

        one_hot_vectors, _ = label_to_one_hot(test_labels)
        test_classes = one_hot_vectors.argmax(axis=-1)

    return pred_classes, test_classes, np.array(pred_confidences)

def predict_lda_traj_classes(temperature: int, scaler: StandardScaler, model,write_to_file=False)-> Dict[int, np.ndarray]:
    """Predict the structure classes for the given test structures.
    The test set contains structures from the isothermal compression of LDA at the given temperature.
    For each pressure there are 10 structures. 
    We take the average Steinhardt descriptor for each pressure and predict the class for that average descriptor.

    Parameters
    ----------
    temperature : int
        The temperature of the isothermal compression.
    scaler : StandardScaler
        The scaler used to standardise the descriptors.
    model : 
        The trained model to use for prediction.
    write_to_file : bool, optional
        Whether to write the predicted classes to file, by default False.
        Use this to generate the colour-coded snapshots shown in Fig. 4 of the paper.

    Returns
    -------
    Dict[int, np.ndarray]
        A dictionary mapping the pressure to the predicted classes.
    """

    pressures =np.arange(100, 20_001, 100)
            
    traj_proportions = {}
    traj_pred_confidences = {} 
    traj_pred_classes = {} 
    for p in pressures:
        file_name = root_dir / f"data/compression_{temperature}K/compression_pressure{p}.extxyz"
        structures = read(file_name, index=':')
        
        # each pressure has 10 structures
        descriptors = []
        for s in structures:
            descriptors.append(s.arrays["steinhardt_descriptor"])
        
        # get a single descriptor for all 10 structures
        # (i.e. the average of all descriptors)
        av_descriptor = np.mean(descriptors, axis=0)

        with torch.no_grad():
            test_x = torch.FloatTensor(scaler.transform(av_descriptor))
            pred_y = torch.nn.Softmax(dim=-1)(model(test_x))

            # get the class with the highest confidence
            pred_class = pred_y.argmax(dim=-1).numpy()
            traj_pred_classes[p] = pred_class
            # get the % confidence for the predicted class
            confidences = pred_y[np.arange(len(pred_y)), pred_class].numpy()
            traj_pred_confidences[p] = confidences
            
            # count the number of 0s, 1s, 2s, and 3s
            class_counts = np.bincount(pred_class, minlength=3)
            traj_proportions[p] = (class_counts / len(pred_class))
            
            # write to file
            # only writes the final structure for each pressure
            # along with the predicted class and confidences
            if write_to_file:
                (
                    hda_confidences,
                    lda_confidences,
                    mda_confidences,
                ) = pred_y.T.numpy()
                s.arrays["predicted_classes"] = pred_class
                s.arrays["prediction_confidence"] = confidences
                s.arrays["hda_confidence"] = hda_confidences
                s.arrays["lda_confidence"] = lda_confidences
                s.arrays["mda_confidence"] = mda_confidences
                write(
                    f"../data/compression_{temperature}K/compression_pressure{p}.extxyz",
                    s,
                    format="extxyz",
                )
        
    return traj_proportions