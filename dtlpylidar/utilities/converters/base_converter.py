import pathlib
from abc import ABC


class BaseToPCDConverter(ABC):
    # Subclasses must define this class variable as default
    extension: str = None
    
    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define the extension attribute."""
        super().__init_subclass__(**kwargs)
        if cls.extension is None:
            raise TypeError(
                f"{cls.__name__} must define 'extension' class variable. "
                f"Example: extension = '.bin'"
            )
    
    def __init__(self, extension=None):
        """
        Initialize converter.
        
        Args:
            extension: Optional extension override. If None, uses class default.
        """
        # Use instance extension if provided, otherwise use class default
        self.extension = extension if extension is not None else self.__class__.extension

    def convert_file(self, input_file, output_file=None, **kwargs):
        raise NotImplementedError("convert_file method is not implemented")

    def convert_folder(self, input_folder, output_folder=None, **kwargs):
        sorted_flag = kwargs.get("sorted", False)

        extension = self.extension
        if not extension.startswith("."):
            extension = f".{extension}"

        data_filepaths = pathlib.Path(input_folder).rglob(f"*{extension}")
        if sorted_flag:
            data_filepaths = sorted(data_filepaths)
        
        output_results = []
        for data_filepath in data_filepaths:
            output_filepath = pathlib.Path(output_folder).joinpath(data_filepath.with_suffix(".pcd").relative_to(input_folder))
            output_result = self.convert_file(data_filepath, output_filepath, **kwargs)
            output_results.append(output_result)
        
        print(f"Successfully converted {len(output_results)} files")
        return output_results