import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget, QMessageBox
from qtpy.QtCore import QEventLoop
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import napari
from magicgui import magicgui
import cupy as cp
from tqdm import tqdm
import warnings
import time
from imageio import get_writer
import tifffile as tf
from scipy.ndimage import zoom
from napari_animation import Animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from skimage import measure

def object_labelling(GP_layer, Image_format):

    all_GPMV_labels = []
    all_GPMV_props = []
    #apply mask to GP_layer: all points that are not nan are set to 1, all points that are nan are set to 0
    GP_layer_mask = np.zeros_like(GP_layer, dtype=bool)
    GP_layer_mask[~np.isnan(GP_layer)] = 1
    
    if Image_format["sizeT"] > 1:
        for t in tqdm(range(GP_layer.shape[0]), desc="Labeling objects", unit="frame", leave=False):
            GPMV_labels = measure.label(GP_layer_mask[t], connectivity=3)
            #remove small regions (less than 100 pixels) from GPMV_labels

            # Remove small regions from labels
            GPMV_props = measure.regionprops(GPMV_labels)
            for prop in GPMV_props:
                if prop.area < 1000:
                    GPMV_labels[GPMV_labels == prop.label] = 0
                
            # Relabel to make indices consecutive
            GPMV_labels = measure.label(GPMV_labels > 0, connectivity=3)
            GPMV_props = measure.regionprops(GPMV_labels)

            if t!=0:
                pass
                tracked_GPMV_labels = np.zeros_like(GPMV_labels, dtype=int)
                previous_GPMV_centroid = np.array([prop.centroid for prop in all_GPMV_props[t-1]])
                current_GPMV_centroid = np.array([prop.centroid for prop in GPMV_props])

                # Calculate distances between centroids of current and previous time points
                distances = np.linalg.norm(current_GPMV_centroid[:, np.newaxis] - previous_GPMV_centroid, axis=2)

                #find the closest previous label for each current label, also store the distance
                closest_previous_labels = np.argmin(distances, axis=1)
                closest_previous_distances = np.min(distances, axis=1)

                # Assign the closest previous label to the current label. If two current labels are close to the same previous label, assign the one with the smallest distance with the previous label. The other gets assigned to a new label
                #find way to remove duplicates
                for i, current_label in enumerate(np.unique(GPMV_labels)):
                    if current_label == 0: # Skip background label
                        continue

                    closest_previous_label = closest_previous_labels[i-1]
                    closest_previous_distance = closest_previous_distances[i-1]

                    #if the closest_previous_distance is higher than prop.feret_diameter_max of the previous label and the prop.area of the current label is outside of the prop.area of the previous label +- 10%, assign a new label
                    if closest_previous_distance > all_GPMV_props[t-1][closest_previous_label].feret_diameter_max and (GPMV_props[i-1].area < all_GPMV_props[t-1][closest_previous_label].area * 0.9 or GPMV_props[i-1].area > all_GPMV_props[t-1][closest_previous_label].area * 1.1):
                        tracked_GPMV_labels[GPMV_labels == current_label] = (np.max(np.unique(GPMV_labels)) + 1)

                    else:
                        tracked_GPMV_labels[GPMV_labels == current_label] = closest_previous_label + 1
                    
                    tracked_GPMV_props = measure.regionprops(tracked_GPMV_labels)

                all_GPMV_labels.append(tracked_GPMV_labels)
                all_GPMV_props.append(tracked_GPMV_props)

            else:
                # Store results for this time point
                all_GPMV_labels.append(GPMV_labels)
                all_GPMV_props.append(GPMV_props)

    else: 
        # For 2D images, use the same logic as above but without time dimension
        GPMV_labels = measure.label(GP_layer_mask, connectivity=3)
        #remove small regions (less than 100 pixels) from GPMV_labels

        # Remove small regions from labels
        GPMV_props = measure.regionprops(GPMV_labels)
        for prop in GPMV_props:
            if prop.area < 100:
                GPMV_labels[GPMV_labels == prop.label] = 0
            
        # Relabel to make indices consecutive
        GPMV_labels = measure.label(GPMV_labels > 0, connectivity=3)
        GPMV_props = measure.regionprops(GPMV_labels)

        all_GPMV_labels.append(GPMV_labels)
        all_GPMV_props.append(GPMV_props)
    
    return all_GPMV_labels, all_GPMV_props

def order_threshold_selector(GP_data, image_format):
    """Ask the user to select a threshold within Napari and block execution until done."""

    #app = QApplication.instance()  # Get existing QApplication if running inside Napari
    event_loop = QEventLoop()  # Create an event loop
    
    viewer = napari.Viewer()
    viewer.add_image(GP_data, name="GP", blending = 'additive')

    viewer.add_image(np.zeros_like(GP_data), name="Ordered region", blending='translucent', opacity=0.5, colormap='I Purple', contrast_limits=[0, 1], multiscale=False)
    viewer.add_image(np.zeros_like(GP_data), name="Disordered region", blending='translucent', opacity=0.5, colormap='I Forest', contrast_limits=[0, 1], multiscale=False)

    viewer.dims.ndisplay = 2  # Set to 2D display

    threshold_parameters = {"threshold_value": None}

    # Create the Matplotlib figure and canvas for histogram
    fig, ax = plt.subplots(figsize=(4, 2))
    canvas = FigureCanvas(fig)
    
    # Plot initial histogram
    ax.hist(GP_data.ravel(), bins=50, color='blue', alpha=0.7)
    ax.set_title("Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    
    # Add vertical line for threshold
    threshold_line = ax.axvline(x=0, color='red', linestyle='--')

    @magicgui(
        threshold_value={"label": "Order threshold", 
                        "widget_type": "FloatSpinBox", 
                        "min": -1, 
                        "max": 1, 
                        "step": 0.01, 
                        "value": 0},
        call_button="Run"
    )
    def parameter_selector(threshold_value: float):
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

    @parameter_selector.threshold_value.changed.connect
    def _on_threshold_change(value: float):
        # Update the vertical line in the histogram
        threshold_line.set_xdata([value, value])
        canvas.draw_idle()

        update_mask()

    def update_mask():
        threshold = parameter_selector.threshold_value.value

        order_mask = GP_data > threshold
        disorder_mask = GP_data <= threshold

        if "Ordered region" in viewer.layers and "Disordered region" in viewer.layers:
            viewer.layers["Ordered region"].data = order_mask
            viewer.layers["Disordered region"].data = disorder_mask

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
    parameter_selector.native.setMinimumWidth(400)
    # **Fix: Explicitly add magicgui as a dock widget**
    dock_widget = viewer.window.add_dock_widget(container, area="right", name="Threshold Selector")

    # **Fix: Allow time for UI to render before starting the event loop**
    viewer.window._qt_window.show()
    event_loop.exec_()  # Blocks execution until user confirms

    event_loop.quit()  # Ensure event loop quits fully before cleanup
    # Clean up matplotlib resources before closing viewer
    canvas.close()
    plt.close(fig)

    viewer.close()

    return threshold_parameters["threshold_value"]

def create_directories(base_path, filename):
    """Create necessary directories for results."""
    dirs = ['Results']
    paths = {}
    for dir_name in dirs:
        dir_path = os.path.join(base_path, filename)
        os.makedirs(dir_path, exist_ok=True)
        paths[dir_name] = dir_path
    return paths

def calculate_GP_histogram(GP_layer, image_format):
    """Calculate the histogram of the GP layer."""
    frame_data = {'global' : [], 'slice' : dict()}

    global_median = np.nanmedian(GP_layer.flatten())
    global_variance = np.nanvar(GP_layer.flatten())
    global_hist, global_bin_edges = np.histogram(GP_layer, bins=256, range=(-1, 1))
    bin_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
    frame_data['global'] = [global_hist, global_bin_edges, bin_centers, global_median, global_variance]

    if image_format["sizeT"] > 1:
        for t in range(image_format["sizeT"]):
            median = np.nanmedian(GP_layer[t].flatten())
            variance = np.nanvar(GP_layer[t].flatten())
            hist, bin_edges = np.histogram(GP_layer[t], bins=256, range=(-1, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            frame_data['slice'][t] = [hist, bin_edges, bin_centers, median, variance]

    else:
        frame_data['slice'] = None
    
    return frame_data

def results_calc(GP_data, image_format, threshold, all_object_labels):
    """Calculate the percentage of ordered region area object per object"""
    unique_labels = np.unique(all_object_labels)

    image_results = {'global': {}, 'slice': {}}

    for label in tqdm(unique_labels, desc="Calculating results", unit="label", leave=False):
        if label == 0: #skip background label
            continue

        order_percentage_list = []
        order_median_list = []
        ordered_pixels_list = []
        disorder_median_list = []
        disordered_pixels_list = []
        image_median_list = []
        image_variance_list = []

        label_mask = all_object_labels == label
        masked_GP = np.where(label_mask, GP_data, np.nan)
        label_order_phase = masked_GP[masked_GP > threshold]
        label_disorder_phase = masked_GP[masked_GP <= threshold]

        #apply the mask on GP_data
        image_median = np.nanmedian(masked_GP)
        image_variance = np.nanvar(masked_GP)
        order_median = np.nanmedian(label_order_phase)
        disorder_median = np.nanmedian(label_disorder_phase)
        #get the number of pixels in the ordered and disordered phase
        
        ordered_pixels = np.count_nonzero(~np.isnan(label_order_phase))
        disordered_pixels = np.count_nonzero(~np.isnan(label_disorder_phase))

        order_percentage = (ordered_pixels / (ordered_pixels + disordered_pixels))*100

        image_results['global'][label] = {"Order percentage": order_percentage,
                                        "Ordered pixels": ordered_pixels,
                                        "Order median": order_median,
                                        "Disordered pixels": disordered_pixels,
                                        "Disorder median": disorder_median,
                                        "Image median": image_median,
                                        "Image variance": image_variance
                                        }

        if image_format['sizeT'] > 1:
            for t in range(image_format["sizeT"]):
                image_median = np.nanmedian(masked_GP[t])
                image_variance = np.nanvar(masked_GP[t])

                order_median = np.nanmedian(label_order_phase[t])
                disorder_median = np.nanmedian(label_disorder_phase[t])

                ordered_pixels = np.sum(label_order_phase[t])
                disordered_pixels = np.sum(label_disorder_phase[t])

                order_percentage = (ordered_pixels / (ordered_pixels + disordered_pixels))*100

                order_median_list.append(order_median)
                disorder_median_list.append(disorder_median)
                image_median_list.append(image_median)
                image_variance_list.append(image_variance)
                ordered_pixels_list.append(ordered_pixels)
                disordered_pixels_list.append(disordered_pixels)
                order_percentage_list.append(order_percentage)

        image_results['slice'][label] = {"Order percentage": order_percentage_list, 
                                "Ordered pixels": ordered_pixels_list,
                                "Order median": order_median_list,
                                "Disordered pixels": disordered_pixels_list,
                                "Disorder median": disorder_median_list,
                                "Image median": image_median_list,
                                "Image variance": image_variance_list
                                }

    return image_results

def plot_frame_scatter(sizeT, frames_medians, frames_variances, order_percentages, order_medians, disordered_medians, order_threshold, filename, save_dir, label):
    """Plot and save the scatter plot of median and variance for all of the frames."""
    # Create figure and primary axis
    x = np.arange(sizeT)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9, 24))

    # Plot median on primary y-axis
    color_median = 'tab:blue'
    color_order = 'tab:green'

    color_order_median = 'xkcd:sky blue'
    color_disorder_median = 'xkcd:royal blue'

    #First Subplot: general scatter plot

    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Median', color=color_median)
    axes[0].scatter(x, frames_medians, color=color_median, label='Median', s=15, alpha=0.7)
    
    #also plot order and disorder median with different markers
    axes[0].scatter(x, order_medians, color=color_order_median, label='Order Median', s=15, marker="+", alpha=0.7)
    axes[0].scatter(x, disordered_medians, color=color_disorder_median, label='Disorder Median', s=15, marker="v", alpha=0.7)

    axes[0].tick_params(axis='y', labelcolor=color_median)
    axes[0].axhline(y=order_threshold, color=color_median, linestyle='--', label='Order Threshold')


    '''
    # Create secondary y-axis for variance
    ax2 = ax1.twinx()
    color_variance = 'tab:red'
    ax2.set_ylabel('Variance', color=color_variance)
    ax2.scatter(x, frames_variances, color=color_variance, label='Variance', s=30, marker='x', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color_variance)
    '''

    # Create secondary y-axis for order percentage
    ax3 = axes[0].twinx()
    ax3.set_ylabel('Order %', color=color_order)
    ax3.scatter(x, order_percentages, color=color_order, label='Order %', s=15, marker='x', alpha=0.7)
    ax3.tick_params(axis='y', labelcolor=color_order)

    #Second Subplot: Global Median scatter plot
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Global Median', color=color_median)
    axes[1].scatter(x, frames_medians, color=color_median, label='Median', s=10, alpha=0.7)
    axes[1].tick_params(axis='y', labelcolor=color_median)
    # Create secondary y-axis for order percentage
    ax4 = axes[1].twinx()
    ax4.set_ylabel('Order %', color=color_order)
    ax4.scatter(x, order_percentages, color=color_order, label='Order %', s=10, marker='x', alpha=0.7)
    ax4.tick_params(axis='y', labelcolor=color_order)

    #Third Subplot: Ordered median scatterplot
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Ordered Median', color=color_median)
    axes[2].scatter(x, order_medians, color=color_order_median, label='Order Median', s=10, marker="+", alpha=0.7)
    axes[2].tick_params(axis='y', labelcolor=color_median)

    #Fourth Subplot: Disordered median scatterplot
    axes[3].set_xlabel('Frame')
    axes[3].set_ylabel('Disordered Median', color=color_median)
    axes[3].scatter(x, disordered_medians, color=color_disorder_median, label='Disorder Median', s=10, marker="v", alpha=0.7)
    axes[3].tick_params(axis='y', labelcolor=color_median)

    # Collect handles and labels from all axes (including twin axes)
    handles_labels = [ax.get_legend_handles_labels() for ax in [axes[0], ax3, axes[1], ax4, axes[2], axes[3]]]
    handles, labels = zip(*handles_labels)

    # Flatten lists and remove duplicates
    all_handles = [h for sublist in handles for h in sublist]
    all_labels = [l for sublist in labels for l in sublist]
    unique = dict(zip(all_labels, all_handles))

    # Place global legend above all subplots
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=10,
        frameon=True
    )


    fig.suptitle(f'Medians and Order % over time', fontsize=16, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust space for legend and title

    save_path = os.path.join(save_dir, f'{filename}_object{label}_median_order_scatterplot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_GP_histogram(hist_df, filename, frame, save_dir):
    """Plot and save the histogram."""
    plt.figure(figsize=(8, 6))
    plt.bar(hist_df['bin_start'], hist_df['count'], width=0.02)
    plt.xlabel('GP Value')
    plt.ylabel('Count')
    if frame == 'global':
        plt.title(f'GP Histogram - {filename} - Global')
        plt.savefig(os.path.join(save_dir, f'{filename}_global_histogram.png'))

    else:
        plt.title(f'GP Histogram - {filename} - Frame {frame}')
        plt.savefig(os.path.join(save_dir, f'{filename}_frame-{frame}_histogram.png'))
    plt.close()

def image_3d_sample(GP_layer, filename, save_dir):
    """Rotates the sample around the x and y axis and captures frames."""
    max_val = np.nanmax(GP_layer)
    min_val = np.nanmin(GP_layer)

    viewer = napari.Viewer(show=True)  # Need to show for rendering to work

    viewer.add_image(GP_layer, name="Greyscale copy", scale=[1, 1, 1], blending='translucent', opacity=1, colormap='gray_r')
    viewer.add_image(GP_layer, name="3D Sample", scale=[1, 1, 1], blending='translucent', opacity=0.6, colormap='hsv', contrast_limits=[min_val, max_val])

    # Set to 3D mode and initial angle
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 0, 35)

    #make axis visible
    viewer.axes.visible = True

    animation = Animation(viewer)

    angle1 = 0
    angle2 = 180
    angle3 = 360

    viewer.camera.angles = (0, angle1, 45)  # Rotate around Y-axis
    animation.capture_keyframe()

    viewer.camera.angles = (0, angle2, 45)  # Rotate around Y-axis
    animation.capture_keyframe()

    viewer.camera.angles = (0, angle3, 45)  # Rotate around Y-axis
    animation.capture_keyframe()


    save_name = os.path.join(save_dir, f'{filename}_3D_sample.mp4')
    # Save as a video
    animation.animate(save_name, fps=10, quality = 10, canvas_only=True, scale_factor=1)
    viewer.close()
    del viewer  # Ensure it gets garbage collected

    return max_val, min_val

def image_4d_sample(GP_layer, filename, sizeT, save_dir):
    
    max_val = np.nanmax(GP_layer)
    min_val = np.nanmin(GP_layer)

    viewer = napari.Viewer(show=True)

    viewer.add_image(GP_layer, name="Greyscale copy", scale=[1, 1, 1], blending='translucent', opacity=1, colormap='gray_r')
    viewer.add_image(GP_layer, name="4D Sample", scale=[1, 1, 1], blending='translucent', opacity=0.6, colormap='hsv', contrast_limits=[min_val, max_val])

    # Set to 3D mode and initial angle
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 0, 35)

    #make axis visible
    viewer.axes.visible = True

    # Create animation
    animation = Animation(viewer)

    angle1 = -30 
    angle2 = 0
    angle3 = 30
    t1 = 0
    t2 = int(sizeT/2)
    t3 = sizeT-1

    viewer.dims.set_point(0, t1)  # Update time axis
    viewer.camera.angles = (0, angle1, 45)  # Rotate around Y-axis
    animation.capture_keyframe()

    viewer.dims.set_point(0, t2)  # Update time axis
    viewer.camera.angles = (0, angle2, 45)  # Rotate around Y-axis
    animation.capture_keyframe()

    viewer.dims.set_point(0, t3)  # Update time axis
    viewer.camera.angles = (0, angle3, 45)  # Rotate around Y-axis
    animation.capture_keyframe()

    save_name = os.path.join(save_dir, f'{filename}_4D_sample.mp4')

    # Save as a video
    animation.animate(save_name, fps=10, quality = 10, canvas_only=True, scale_factor=1)

    viewer.close()
    del viewer  # Ensure it gets garbage collected

    return max_val, min_val

def image_2d_tseries(GP_layer, filename, sizeT, save_dir):
    max_val = np.nanmax(GP_layer)
    min_val = np.nanmin(GP_layer)

    viewer = napari.Viewer(show=True)

    viewer.add_image(GP_layer, name="Greyscale copy", scale=[1, 1, 1], blending='translucent', opacity=1, colormap='gray_r')
    viewer.add_image(GP_layer, name="2D Time Series", scale=[1, 1, 1], blending='translucent', opacity=0.6, colormap='hsv', contrast_limits=[min_val, max_val])

    viewer.dims.ndisplay = 2
    viewer.camera.angles = ( 0, 35)

    #make axis visible
    viewer.axes.visible = True

    # Create animation
    animation = Animation(viewer)

    t1 = 0
    t2 = int(sizeT/2)
    t3 = sizeT-1

    viewer.dims.set_point(0, t1)  # Update time axis
    animation.capture_keyframe()

    viewer.dims.set_point(0, t2)  # Update time axis
    animation.capture_keyframe()

    viewer.dims.set_point(0, t3)  # Update time axis
    animation.capture_keyframe()
    
    save_name = os.path.join(save_dir, f'{filename}_2D_timeseries.mp4')

    # Save as a video
    animation.animate(save_name, fps=10, quality = 10, canvas_only=True, scale_factor=1)

    viewer.close()
    del viewer  # Ensure it gets garbage collected

    return max_val, min_val

def colormap_scale(min_val, max_val, save_dir):
    """Create a scale bar for the colormap."""
    cmap = 'hsv'
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    gradient = np.linspace(min_val, max_val, 256).reshape(1, -1)
    #limit min and max_val to 2 decimals

    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[min_val, max_val, 0, 1])

    ax.set_xticks(np.linspace(min_val, max_val, 5))
    ax.set_yticks([])
    ax.set_title(f'Scale Bar ({cmap} colormap)')

    save_path = os.path.join(save_dir, 'colormap_scale.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_results(output_path, filename, GP_layer, image_format, sum_threshold, order_threshold, all_object_labels, all_object_props):
    """Save results, histograms, and images."""
    paths = create_directories(output_path, filename)
    image_results = results_calc(GP_layer, image_format, order_threshold, all_object_labels)
    unique_labels = np.unique(all_object_labels)
    
    slice_df_list = []
    global_df_list = []
    for label in unique_labels:
        if label == 0: # Skip background label
            continue
        if image_format["sizeT"] > 1:

            frames = np.arange(image_format["sizeT"]).squeeze()
            label_name = np.full(image_format["sizeT"], str(label))
            frames_medians = np.array([image_results['slice'][label]['Image median']]).squeeze()
            frames_variances = np.array([image_results['slice'][label]['Image variance']]).squeeze()
            order_percentages = np.array([image_results['slice'][label]['Order percentage']]).squeeze()
            order_medians = np.array([image_results['slice'][label]['Order median']]).squeeze()
            ordered_pixels = np.array([image_results['slice'][label]['Ordered pixels']]).squeeze()
            disordered_medians = np.array([image_results['slice'][label]['Disorder median']]).squeeze()
            disordered_pixels = np.array([image_results['slice'][label]['Disordered pixels']]).squeeze()
            volumes = np.array([all_object_props[t][label-1].area for t in range(image_format["sizeT"])]).squeeze()
            centroids = [tuple(all_object_props[t][label-1].centroid) for t in range(image_format["sizeT"])]
            centroids_x = np.array([c[0] for c in centroids])
            centroids_y = np.array([c[1] for c in centroids])
            centroids_z = np.array([c[2] for c in centroids])
            ferets = np.array([all_object_props[t][label-1].feret_diameter_max for t in range(image_format["sizeT"])]).squeeze()

            GP_results_df = pd.DataFrame({'frames': frames,
                                            'Label': label_name,
                                            'Median': frames_medians, 
                                            'Variance': frames_variances,
                                            'Order_percentages': order_percentages,
                                            'Order_median': order_medians,
                                            'Ordered_pixels_nb': ordered_pixels,
                                            'Disordered_median': disordered_medians,
                                            'Disordered_pixels_nb': disordered_pixels,
                                            'Volume': volumes,
                                            'Centroid_x': centroids_x,
                                            'Centroid_y': centroids_y,
                                            'Centroid_z': centroids_z,
                                            'Feret_diameter': ferets,
                                            })
            
            slice_df_list.append(GP_results_df)
            
            plot_frame_scatter(image_format['sizeT'], 
                            frames_medians, 
                            frames_variances, 
                            order_percentages, 
                            order_medians, 
                            disordered_medians,
                            order_threshold,
                            filename, 
                            paths['Results'],
                            label)
            volume = np.nan
            centroid_x = np.nan
            centroid_y = np.nan
            centroid_z = np.nan
            feret = np.nan
        
        else: 
            #if one frame add info about volume , centroid and feret diameter
            volume = np.array([all_object_props[0][label-1].area]).squeeze()
            centroid_x = np.array([all_object_props[0][label-1].centroid[0]]).squeeze()
            centroid_y = np.array([all_object_props[0][label-1].centroid[1]]).squeeze()
            centroid_z = np.array([all_object_props[0][label-1].centroid[2]]).squeeze()
            feret = np.array([all_object_props[0][label-1].feret_diameter_max]).squeeze()
        

        global_GP_results_df = pd.DataFrame({'label': [label],
                                            'Median': [image_results['global'][label]['Image median']],
                                            'Variance': [image_results['global'][label]['Image variance']],
                                            'Sum_threshold': [sum_threshold],
                                            'Order_threshold': [order_threshold],
                                            'Order_percentage': [image_results['global'][label]['Order percentage']],
                                            'Ordered_pixels_nb': [image_results['global'][label]['Ordered pixels']],
                                            'Order_median': [image_results['global'][label]['Order median']],
                                            'Disordered_pixels_nb': [image_results['global'][label]['Disordered pixels']],
                                            'Disorder_median': [image_results['global'][label]['Disorder median']],
                                            'Volume': [volume],
                                            'Centroid_x': [centroid_x],
                                            'Centroid_y': [centroid_y],
                                            'Centroid_z': [centroid_z],
                                            'Feret_diameter': [feret]
                                            })

        global_df_list.append(global_GP_results_df)

    #concatenate all dataframes and save them to csv
    global_GP_results_df = pd.concat(global_df_list, axis=0)
    global_GP_results_df.to_csv(os.path.join(paths['Results'], f'{filename}_global_GP_results.csv'))
    del global_GP_results_df, image_results  # Free up memory

    if image_format["sizeT"] > 1:
        slice_GP_results_df = pd.concat(slice_df_list, axis=0)
        slice_GP_results_df.to_csv(os.path.join(paths['Results'], f'{filename}_frames_GP_results.csv'))
        del frames_medians, frames_variances, order_percentages, order_medians, disordered_medians, ordered_pixels, disordered_pixels, volumes, centroids, ferets # Free up memory

    #save GP layer
    tf.imwrite(os.path.join(paths['Results'], f'{filename}_GP_layer.tif'), GP_layer)
    #save object labels
    tf.imwrite(os.path.join(paths['Results'], f'{filename}_label_masks.tif'), np.array(all_object_labels).astype(np.uint16))

    # Suppress warnings from PyQt5 related to font issues
    warnings.filterwarnings("ignore", category=UserWarning, message=".*DirectWrite.*")
    warnings.filterwarnings("ignore", 
    message=".*input image is not divisible by macro_block_size.*", 
    category=UserWarning, 
    module='imageio')

    #Start rotating and capturing frames
    if image_format["sizeT"] == 1 and image_format["sizeZ"] > 1:
        max_val, min_val = image_3d_sample(GP_layer, filename, paths['Results'])
        print('3D sample recorded')
    
    if image_format["sizeT"] > 1 and image_format["sizeZ"] > 1:
        max_val, min_val = image_4d_sample(GP_layer, filename, image_format['sizeT'], paths['Results'])
        print('4D sample recorded')
    

    if image_format["sizeT"] > 1 and image_format["sizeZ"] == 1:
        max_val, min_val = image_2d_tseries(GP_layer, filename, image_format['sizeT'], paths['Results'])
        print('2D timeseries recorded')

    colormap_scale(min_val, max_val, paths['Results'])

    return min_val, max_val

def final_visualization(GP_layer,
                        order_threshold, 
                        image_format,
                        all_object_labels, 
                        min_val,
                        max_val):
    """Visualize the final results in Napari."""

    order_mask = GP_layer > order_threshold
    disorder_mask = GP_layer <= order_threshold

    viewer = napari.Viewer()
    
    viewer.add_image(GP_layer, name="GP_layer", blending='translucent', opacity=0.5, colormap='hsv', multiscale=False, contrast_limits=[min_val, max_val])

    viewer.add_image(order_mask, name="Ordered region", blending='translucent', opacity=0.5, colormap='yellow', visible=False, contrast_limits=[0,1], multiscale=False)
    viewer.add_image(disorder_mask, name="Disordered region", blending='translucent', opacity=0.5, colormap='bop purple', visible=False, contrast_limits=[0,1], multiscale=False)
    viewer.add_labels(np.array(all_object_labels), name='Object Labels', visible = False)
    viewer.axes.visible = True  # Show axis bars
    if image_format["sizeZ"] > 1:
        viewer.dims.axis_labels = ["Z", "Y", "X"]
        viewer.dims.ndisplay = 3
    
    else:
        viewer.dims.axis_labels = ["Y", "X"]
        viewer.dims.ndisplay = 2

    napari.run()


