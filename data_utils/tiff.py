import numpy as np
import torch
from tifffile import TiffFile
import tifffile

import os


def tiff_to_array(filename: str) -> torch.Tensor:
    """
        Reads a .tif file and convert it to a torch tensor

        Arguments:
            filename(str): The path to the .tif file

        Returns:
            np.ndarray: A numpy array which contains the pixel values, has a shape (frame, width, height)
    """
    
    ## Reads the full tiff file, output has shape (frame, height, width)
    result = tifffile.imread(filename)

    result = torch.from_numpy(result)
    result = result.to(torch.float32).squeeze()

    return result


def array_to_tiff(filename: str, frames: torch.Tensor | np.ndarray, compress: bool = True, overwrite: bool = False) -> None:
    """
        Writes frames to a tifffile. 

        Arguments:
            filename(str): 
            frames(tensor): Tensor that contains the frames to be written. The framecount is expected to be in the first dimension.
            compress(bool): Boolean that indicates if the resulting file should be compressed. If compression is used, it will use a lossless 
                compression scheme.
    """
    if not overwrite and os.path.exists(filename):
        raise ValueError(f'The file {filename} already exists')

    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()

    tifffile.imwrite(
        filename,
        frames,
        bigtiff=False,
        photometric=tifffile.PHOTOMETRIC.MINISBLACK,
        planarconfig=tifffile.PLANARCONFIG.CONTIG,
        compression=tifffile.COMPRESSION.LZW if compress else tifffile.COMPRESSION.NONE,
    )

def write_tiff(filename, mov):
    # Assert checks similar to MATLAB
    assert len(mov.shape) in [2, 3], 'The data you are trying to save has an unexpected dimension'
    
    # Determine bit depth based on the data type of 'mov'
    dtype_to_bit = {
        np.uint8: 8,
        np.uint16: 16,
        np.uint32: 32
    }
    bit = dtype_to_bit.get(mov.dtype.type, None)
    
    assert bit is not None, 'Unexpected data type for saving TIFF, expected uint8, uint16, or uint32'

    # Ensure the mov is 3D (even if only 1 frame)
    if len(mov.shape) == 2:
        mov = np.expand_dims(mov, axis=-1)  # Add third dimension for single-frame

    # Open a TiffWriter object to write to the file
    with tifffile.TiffWriter(filename) as tif_writer:
        options = {
            'photometric': tifffile.PHOTOMETRIC.MINISBLACK,
            'planarconfig': tifffile.PLANARCONFIG.CONTIG,
            'bitspersample': bit,
        }
        
        # Write first frame
        tif_writer.write(mov[:,:,0], dtype=f'uint{bit}', **options)

        # Write remaining frames if present
        for i in range(1, mov.shape[2]):
            tif_writer.write(mov[:,:,i], dtype=f'uint{bit}', **options)

def tiff_page_count(filename: str) -> int:
    with tifffile.TiffFile(filename) as tif:
        sum = 0
        for page in tif.pages:
            if page.shape:
                sum += page.shape[0]
        return sum