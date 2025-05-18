import numpy as np
from aicsimageio import AICSImage
from tkinter import Tk, filedialog
import xml.etree.ElementTree as ET

def select_path():
    """Open a file dialog to select a file."""
    root = Tk()
    root.title("File Selection")
    root.attributes('-alpha', 0.0)

    root.update_idletasks()
    root.deiconify()
    root.focus_force()  # Force focus on the root window

    file_path = filedialog.askopenfilename(parent = root, title="Select a CZI File", filetypes=[("CZI files", "*.czi")])
    root.destroy()  # Close the root window

    if file_path is None:
        print("No file selected. Exiting...")
        raise SystemExit

    return file_path

def select_path_list():
    """Open a file dialog to select a file."""
    root = Tk()
    root.title("File Selection")
    root.attributes('-alpha', 0.0)

    root.update_idletasks()
    root.deiconify()
    root.focus_force()  # Force focus on the root window

    file_list = filedialog.askopenfilenames(parent = root, title="Select a CZI File", filetypes=[("CZI files", "*.czi")])
    root.destroy()  # Close the root window

    return file_list

def load_czi_file(file_path=None):
    """Load a CZI file and return the image data and metadata XML."""
    if file_path is None:
        file_path = select_path()
        if not file_path:
            raise SystemExit

    img = AICSImage(file_path)
    data = img.get_image_data()  # Adjust dimensions as needed
    data = np.squeeze(data)  # Remove single-dimensional entries
    metadata_xml = img.metadata
    channel_names = []
    channel_emission = []

    for channel in metadata_xml.findall(".//Dimensions/Channels/Channel"):
        name = channel.attrib.get("Name", "")
        channel_names.append(name)

        # Find the DyeMaxEmission element
        emission = channel.find("EmissionWavelength")
        channel_emission.append(emission.text)

        
    sizeT = int(img.dims.T)
    sizeZ = int(img.dims.Z)
    sizeC = int(img.dims.C)

    image_format = {"sizeT": sizeT, "sizeZ": sizeZ, "sizeC": sizeC, "channel_names": channel_names, "channel_emission": channel_emission}

    #metadata_str = ET.tostring(metadata_xml, encoding="unicode", method="xml")

    print(f"Loaded file: {file_path}")

    return data, image_format, file_path