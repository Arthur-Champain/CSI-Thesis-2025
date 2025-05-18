import numpy as np
import cupy as cp
import napari
from magicgui import magicgui
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget, QMessageBox
from qtpy.QtCore import QEventLoop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.filters import threshold_otsu
import gc
from tqdm import tqdm
from scipy.ndimage import median_filter

def sum_threshold_selector(blue_layer, red_layer, image_format):
    """Ask the user to select a threshold within Napari and block execution until done."""
    sum_layer = blue_layer + red_layer
    #bluemax_value = np.max(blue_layer)
    #redmax_value = np.max(red_layer)
    #remove saturated pixels
    sum_max_value = np.max(sum_layer)

    #app = QApplication.instance()  # Get existing QApplication if running inside Napari
    event_loop = QEventLoop()  # Create an event loop
    
    viewer = napari.Viewer()
    viewer.add_image(sum_layer, name="Sum Layer")


    viewer.add_image(np.zeros_like(sum_layer), name="Mask", blending='additive', opacity=0.5, colormap='green', contrast_limits=[0,1], multiscale=False)

    viewer.dims.ndisplay = 2  # Set to 2D display

    threshold_parameters = {"threshold_mode": None, "threshold_value": None}

    # Create the Matplotlib figure and canvas for histogram
    fig, ax = plt.subplots(figsize=(4, 2))
    canvas = FigureCanvas(fig)
    
    # Plot initial histogram
    ax.hist(sum_layer.ravel(), bins=50, color='blue', alpha=0.7)
    ax.set_title("Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    
    # Add vertical line for threshold
    threshold_line = ax.axvline(x=0, color='red', linestyle='--')

    @magicgui(
        threshold_mode={"label": "Threshold Mode", 
                        "choices": [None, "Manual", "Otsu"]},

        threshold_value={"label": "Threshold Value", 
                        "widget_type": "SpinBox", 
                        "min": 0, 
                        "max": int(sum_max_value), 
                        "step": max(int(sum_max_value/1000), 10), 
                        "value": int(sum_max_value/2), 
                        "visible": False},

        call_button="Run"
    )
    def parameter_selector(threshold_mode: str, threshold_value: int):
        if threshold_mode is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please choose a threshold mode.")
            msg.setWindowTitle("Threshold Selection Error")
            msg.exec_()
            return
        threshold_parameters["threshold_mode"] = threshold_mode
        threshold_parameters["threshold_value"] = threshold_value
        event_loop.quit()  # Stop blocking execution

    # Create a container widget to hold both the parameter selector and histogram
    container = QWidget()
    layout = QVBoxLayout()
    
    # Add the magicgui widget to the layout
    layout.addWidget(parameter_selector.native)
    
    # Add the matplotlib canvas to the layout
    layout.addWidget(canvas)
    
    container.setLayout(layout)

    @parameter_selector.threshold_mode.changed.connect
    def _on_mode_change(value: str):
        parameter_selector.threshold_value.visible = (value == "Manual")
        update_mask(value)

    @parameter_selector.threshold_value.changed.connect
    def _on_threshold_change(value: float):
        if parameter_selector.threshold_mode.value == "Manual":
            update_mask("Manual")
            parameter_selector.threshold_value.parent.update()

    def update_mask(mode):
        if mode == "Otsu":
            threshold = threshold_otsu(sum_layer[sum_layer > 0])
        else:
            threshold = parameter_selector.threshold_value.value

        mask = sum_layer > threshold
        if "Mask" in viewer.layers:
            viewer.layers["Mask"].data = mask

        threshold_line.set_xdata([threshold, threshold])
        canvas.draw_idle()

    # Close event: Quit event loop if user closes Napari manually
    def on_close(event):
        # Properly clean up to avoid Qt C++ object deleted error
        try:
            viewer.window.remove_dock_widget(dock_widget)
        except:
            pass
        canvas.close()
        plt.close(fig)
        event_loop.quit()

    viewer.window._qt_window.closeEvent = on_close  # Capture close event

    # **Fix: Explicitly add magicgui as a dock widget**
    dock_widget = viewer.window.add_dock_widget(container, area="right", name="Threshold Selector")
    #parameter_selector.native.setMinimumWidth(300)  # Adjust width as needed

    # **Fix: Allow time for UI to render before starting the event loop**
    viewer.window._qt_window.show()
    event_loop.exec_()  # Blocks execution until user confirms

    event_loop.quit()  # Ensure event loop quits fully before cleanup
    # Clean up matplotlib resources before closing viewer
    canvas.close()
    plt.close(fig)

    viewer.close()

    if threshold_parameters["threshold_mode"] == "Manual":
        threshold_parameters["threshold_value"] = threshold_parameters["threshold_value"]
    elif threshold_parameters["threshold_mode"] == "Otsu":
        threshold_parameters["threshold_value"] = threshold_otsu(sum_layer[sum_layer > 0])



    return threshold_parameters["threshold_value"]

def GP_calculation(blue_layer, red_layer, image_format, chunk_size=10000):
    gc.collect()
    

    sum_threshold = sum_threshold_selector(blue_layer, red_layer, image_format)
    if sum_threshold is None:
            raise SystemExit
            

    blue = cp.asarray(blue_layer).astype(cp.float32)
    red = cp.asarray(red_layer).astype(cp.float32)
    del blue_layer, red_layer # Free up memory

    sum_layer = blue + red
    numerator = blue - red

    del blue, red  # Free up memory
    cp.get_default_memory_pool().free_all_blocks()  # Release memory back to GPU
    
    GP_layer = cp.full_like(sum_layer, cp.nan, dtype=cp.float32)

    # Create mask and calculate GP
    mask = sum_layer > sum_threshold
    GP_layer[mask] = cp.divide(numerator[mask], sum_layer[mask])

    # Median filter the GP layer
    GP_layer_medFilt = np.full_like(GP_layer.shape, np.nan, dtype=np.float32)
    if image_format['sizeT'] > 1:
        for t in tqdm(range(image_format['sizeT']), desc="Median Filtering", unit="frame", leave=False):
            GP_layer[t] = median_filter(GP_layer[t], size=2)
    
    else:
        GP_layer[t] = median_filter(GP_layer, size=2)
            
    return GP_layer_medFilt, sum_threshold

def GP_calculation_chunked(blue_layer, red_layer, image_format, sum_threshold = None, chunk_size=1000):
    gc.collect()
    
    if sum_threshold is None:
        sum_threshold = sum_threshold_selector(blue_layer, red_layer, image_format)
    
    if sum_threshold is None:
        raise SystemExit
    
    # Create output array on CPU initially
    result_shape = blue_layer.shape
    GP_layer_cpu = np.full(result_shape, np.nan, dtype=np.float32)
    
    # Calculate total number of elements and determine chunking
    total_pixels = blue_layer.size
    chunk_size = total_pixels // 10  # 10 chunks
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Process data in chunks
    for i in tqdm(range(num_chunks), desc="Calculating GP", unit="chunk", leave=False):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_pixels)
        
        # Get slice indices for the original shape
        indices = np.unravel_index(np.arange(start_idx, end_idx), result_shape)
        
        # Extract chunk data
        blue_chunk = blue_layer[indices].astype(np.float32)
        red_chunk = red_layer[indices].astype(np.float32)
        
        # Transfer chunks to GPU
        blue_gpu = cp.asarray(blue_chunk)
        red_gpu = cp.asarray(red_chunk)
        
        # Process on GPU
        sum_chunk = blue_gpu + red_gpu
        numerator_chunk = blue_gpu - red_gpu
        
        # Free intermediate GPU memory
        del blue_gpu, red_gpu
        
        # Create mask and calculate GP for this chunk
        mask = sum_chunk > sum_threshold
        GP_chunk = cp.full_like(sum_chunk, cp.nan, dtype=cp.float32)
        GP_chunk[mask] = cp.divide(numerator_chunk[mask], sum_chunk[mask])
        
        # Transfer results back to CPU
        GP_layer_cpu[indices] = cp.asnumpy(GP_chunk)
        
        # Free GPU memory
        del sum_chunk, numerator_chunk, GP_chunk
        cp.get_default_memory_pool().free_all_blocks()

    # Free up memory
    del blue_layer, red_layer

    if image_format['sizeT'] > 1:
        for t in tqdm(range(image_format['sizeT']), desc="Median Filtering", unit="frame", leave=False):
            GP_layer_cpu[t] = median_filter(GP_layer_cpu[t], size=2)
    
    else:
        GP_layer_cpu = median_filter(GP_layer_cpu, size=2)
    
    return GP_layer_cpu, sum_threshold