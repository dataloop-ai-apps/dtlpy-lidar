import os
import pathlib
import dtlpy as dl
import json
import urllib.request
import urllib.error
import zipfile
import logging
import shutil

logger = logging.getLogger(name='dataset-uploader')


class DatasetUploader(dl.BaseServiceRunner):
    def __init__(self, source: str, download_path: str = None, ontology_filepath: str = None):
        """
        Initialize the DatasetUploader
        :param source: Required: Source URL to download the dataset content as zip
        :param download_path: Path to the downloaded dataset content
        :param ontology_filepath: Path to the ontology file
        """
        super().__init__()
        self.source = source
        self.download_path = download_path if download_path else os.getcwd()
        self.ontology_filepath = ontology_filepath if ontology_filepath else str(pathlib.Path(os.getcwd()) / "ontology.json")

    @staticmethod
    def import_recipe_ontology(dataset: dl.Dataset, ontology_filepath: str):
        """
        Import the recipe ontology to the dataset
        :param dataset: Dataset to import the recipe ontology to
        :param ontology_filepath: Path to the ontology file
        :return: Ontology object
        """
        recipe: dl.Recipe = dataset.recipes.list()[0]
        ontology: dl.Ontology = recipe.ontologies.list()[0]
        if pathlib.Path(ontology_filepath).exists():
            with open(file=ontology_filepath, mode='r') as file:
                new_ontology_json = json.load(fp=file)
            new_ontology = ontology.copy_from(ontology_json=new_ontology_json)
        else:
            logger.warning(f"Ontology file not found at: {ontology_filepath}, ontology will not be updated.")
            new_ontology = ontology
        return new_ontology

    @staticmethod
    def download_zip(source: str, download_path: str, progress: dl.Progress = None, **kwargs):
        """
        Download the zip file from the source URL to the download path
        :param source: Source URL to download the dataset content as zip
        :param download_path: Path to the downloaded dataset content
        :param progress: Progress object to update the progress
        :param kwargs: Additional keyword arguments
        :return: Path to the downloaded dataset content
        """
        zip_name = pathlib.Path(source).name
        zip_filepath = pathlib.Path(download_path) / zip_name
        zip_content_path = str(zip_filepath.with_suffix(''))
        if progress is not None:
            progress.update(progress=10, message=f"Downloading dataset from: '{source}' to '{download_path}'")
        # Clean zip content path (to remove any previous extraction)
        shutil.rmtree(path=zip_content_path, ignore_errors=True)
        os.makedirs(name=zip_content_path, exist_ok=True)
        try:
            urllib.request.urlretrieve(url=source, filename=zip_filepath)
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Error downloading data: {e}")
        # Check if zip file is valid
        if not pathlib.Path(zip_filepath).is_file() or not str(zip_filepath).endswith(".zip"):
            raise FileNotFoundError(f"Error: '{zip_filepath}' is not a valid zip file.")
        # Extract zip contents to the zip content path (/<zip_name>.zip -> /<zip_name>)
        zip_ref = zipfile.ZipFile(zip_filepath, 'r')
        zip_ref.extractall(path=zip_content_path)
        zip_ref.close()
        logger.info(f"Extracted contents of '{zip_filepath}' to {zip_content_path}.")
        return zip_content_path

    @staticmethod
    def update_frames_info(frames_item: dl.Item, export_path: str):
        """
        Update the frames item information
        :param frames_item: Frames item to update
        :param export_path: Path to the exported dataset content
        :return: Frames item
        """
        frames_item_json = json.load(fp=frames_item.download(save_locally=False))
        download_path = frames_item.dataset.download_annotations(
            local_path=export_path,
            annotation_options=dl.ViewAnnotationOptions.JSON,
            overwrite=True
        )
        # Update frames item information
        download_json_path = pathlib.Path(download_path) / "json"
        for frame in frames_item_json['frames']:
            frame_remote_path = pathlib.Path(frame['lidar']['remote_path'].lstrip('/')).with_suffix('.json')
            frame_json_path = str(download_json_path / frame_remote_path)
            with open(frame_json_path, 'r') as fp:
                frame_json = json.load(fp=fp)
            frame['lidar']['lidar_pcd_id'] = frame_json['id']
            for image in frame['images']:
                image_remote_path = pathlib.Path(image['remote_path'].lstrip('/')).with_suffix('.json')
                image_json_path = str(download_json_path / image_remote_path)
                with open(image_json_path, 'r') as fp:
                    image_json = json.load(fp=fp)
                image['image_id'] = image_json['id']
        # Upload new frames item
        frames_item.delete()
        frames_item = dataset.items.upload(
            remote_name=frames_item.name,
            remote_path=frames_item.dir,
            local_path=json.dumps(frames_item_json).encode(),
            item_metadata={
                "system": {
                    "shebang": {
                        "dltype": "PCDFrames"
                    }
                },
                "fps": 1
            }
        )
        return frames_item

    @staticmethod
    def upload_data(dataset: dl.Dataset, content_path: str, sequence_name: str, progress: dl.Progress = None, **kwargs):
        """
        Upload the source data to the dataset
        :param dataset: Dataset to upload the source data to
        :param content_path: Path to the source data
        :param sequence_name: Name of the sequence
        :param progress: Progress object to update the progress
        :param kwargs: Additional keyword arguments
        :return: Frames item
        """
        if progress is not None:
            progress.update(progress=40, message="Uploading source data...")
        # Add progress update callback
        progress_tracker = {'last_progress': 0}
        def progress_callback(**kwargs):
            p = kwargs.get('progress')  # p is between 0-100
            if progress is not None:
                progress_int = round(p / 10) * 10  # round to 10th
                if progress_int % 10 == 0 and progress_int != progress_tracker['last_progress']:
                    progress.update(progress=40 + (40 * progress_int / 100), message="Uploading source data...")
                    progress_tracker['last_progress'] = progress_int
        dl.client_api.callbacks.add(event='itemUpload', func=progress_callback)
        # Upload scene folder
        scene_path = os.path.join(content_path, sequence_name)
        dataset.items.upload(local_path=scene_path)
        frames_item = dataset.items.get(filepath=f"/{sequence_name}/frames.json")
        frames_item = DatasetUploader.update_frames_info(frames_item=frames_item, export_path=content_path, update_metadata=True)
        return frames_item

    @staticmethod
    def upload_annotations(frames_item: dl.Item, content_path: str, sequence_name: str, progress: dl.Progress = None, **kwargs):
        """
        Upload the annotations to the frames item
        :param frames_item: Frames item to upload the annotations to
        :param content_path: Path to the source data
        :param sequence_name: Name of the sequence
        :param progress: Progress object to update the progress
        :param kwargs: Additional keyword arguments
        :return: Frames item
        """
        if progress is not None:
            progress.update(progress=90, message="Uploading annotations...")
        # Load annotations and modify them
        annotations_filepath = os.path.join(content_path, f'{sequence_name}_frames.json')
        builder = dl.AnnotationCollection.from_json_file(filepath=annotations_filepath)
        # TODO: Segmentation TBD [START]
        # # Get and Upload Segmentation References
        # dataset = dl.datasets.get(dataset_id=frames_item.dataset.id)
        # sem_ref_path = os.path.join(path, 'sem_ref')
        # sem_ref_items = dataset.items.upload(local_path=sem_ref_path, remote_path="/.dataloop")
        # sem_ref_items_map = {pathlib.Path(item.filename).stem.split('_', 1)[1]: item for item in sem_ref_items}
        # annotation: dl.Annotation
        # for annotation in builder.annotations:
        #     if annotation.type == "ref_semantic_3d":
        #         ref_item = sem_ref_items_map[annotation.label]
        #         annotation.coordinates["ref"] = ref_item.id
        # TODO: Segmentation TBD [END]
        # Remove all segmentation annotations (After segmentation supported - remove this)
        cubes_annotations = list()
        for annotation in builder.annotations:
            if annotation.type == dl.AnnotationType.CUBE3D:
                cubes_annotations.append(annotation)
        builder.annotations = cubes_annotations
        # Upload annotations
        builder.item = frames_item
        builder.upload()

    def upload_dataset(self, dataset: dl.Dataset, sequence_name: str = None, progress: dl.Progress = None, download_zip: bool = True, **kwargs) -> dl.Item:
        """
        The main function to upload the dataset
        :param dataset: Dataset to upload the dataset to
        :param sequence_name: Name of the sequence
        :param progress: Progress object to update the progress
        :param download_zip: Whether to download the zip file
        :param kwargs: Additional keyword arguments
        :return: Frames item
        """
        self.import_recipe_ontology(dataset=dataset, ontology_filepath=self.ontology_filepath)
        if download_zip:
            content_path = self.download_zip(source=self.source, download_path=self.download_path, progress=progress, **kwargs)
        else:
            zip_name = pathlib.Path(self.source).name
            zip_filepath = pathlib.Path(self.download_path) / zip_name
            content_path = str(zip_filepath.with_suffix(''))
        # If sequence name is not provided, use the basename of the content path
        if sequence_name is None:
            sequence_name = pathlib.Path(content_path).name
        frames_item = self.upload_data(dataset=dataset, content_path=content_path, sequence_name=sequence_name, progress=progress, **kwargs)
        self.upload_annotations(frames_item=frames_item, content_path=content_path, sequence_name=sequence_name, progress=progress, **kwargs)
        # Clean up the content path if the zip file was downloaded
        if download_zip:
            shutil.rmtree(path=content_path, ignore_errors=True)
        return frames_item
