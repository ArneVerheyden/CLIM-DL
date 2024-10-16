import numpy as np
from tifffile import TiffFile
import tifffile

def tiff_to_array(filename: str) -> np.ndarray:
    """
        Reads a .tif file and convert it to a numpy array

        Arguments:
            filename(str): The path to the .tif file

        Returns:
            np.ndarray: A numpy array which contains the pixel values, has a shape (width, height, frames)
    """
    
    ## Reads the full tiff file, output has shape (frame, height, width)
    image = tifffile.imread(filename)
    image = np.astype(image, np.float32)
    ## Transpose to shape (width, height, frame)
    return np.transpose(image, [2, 1, 0])

def array_to_tiff(filename: str, frames: np.ndarray) -> None:
    
    tifffile.imwrite(
        filename,
        frames,
        bigtiff=False,
        photometric=tifffile.PHOTOMETRIC.MINISBLACK,
        planarconfig=tifffile.PLANARCONFIG.CONTIG,
        compression=tifffile.COMPRESSION.NONE,
    )

    raise NotImplementedError()


