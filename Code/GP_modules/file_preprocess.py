import napari
import napari.settings
import numpy as np
from magicgui import magicgui
from qtpy.QtWidgets import QApplication, QMessageBox
from qtpy.QtCore import QEventLoop

def specify_channel_crop(data, image_format):
    """Ask the user to select blue and red channels within Napari and block execution until done."""
    ''' 
    app = QApplication.instance()
    if app is None:
        app = QApplication([])  # Create QApplication if not already running
    '''

    event_loop = QEventLoop()  # Create a new event loop for each iteration
    
    napari.settings.get_settings().application.window_maximized = True

    viewer = napari.Viewer()

    

    layer_nb = image_format["sizeC"]
    stack_nb = image_format["sizeZ"]
    length = image_format["sizeT"]

    if image_format["sizeT"] > 1 and image_format["sizeZ"] > 1:
        for c in range(layer_nb):
            viewer.add_image(data[int(length/2), c, :, :, :], 
                            name=f"{c+1} ({image_format['channel_names'][c]}, {image_format['channel_emission'][c]}nm)", blending='translucent', 
                            multiscale=False)
        viewer.add_image(np.zeros_like(data[int(length/2), 0, :, :, :]), name="Crop limits", opacity=0.5, blending='translucent', contrast_limits=[0,1], multiscale=False)
        viewer.dims.ndisplay = 3

    elif image_format["sizeT"] == 1 and image_format["sizeZ"] > 1:
        for c in range(layer_nb):
            viewer.add_image(data[c, :, :, :], 
                            name=f"{c+1} ({image_format['channel_names'][c]}, {image_format['channel_emission'][c]}nm)", 
                            opacity=0.5, blending='translucent', multiscale=False)
        viewer.add_image(np.zeros_like(data[0, :, :, :]), name="Crop limits", opacity=0.5, blending='translucent', contrast_limits=[0,1], multiscale=False)
        viewer.dims.ndisplay = 3
    
    elif image_format["sizeT"] > 1 and image_format["sizeZ"] == 1:
        for c in range(layer_nb):
            viewer.add_image(data[int(length/2), c, :, :], 
                            name=f"{c+1} ({image_format['channel_names'][c]}, {image_format['channel_emission'][c]}nm)", 
                            opacity=0.5, blending='translucent', multiscale=False)
        viewer.add_image(np.zeros_like(data[int(length/2), 0, :, :]), name="Crop limits", opacity=0.5, blending='translucent', contrast_limits=[0,1], multiscale=False)
        viewer.dims.ndisplay = 2

    selected_channels = {"blue_channel": None, "red_channel": None}
    stack_range = {"start": None, "end": None}
    cropping_method = {"cropping": None}

    @magicgui(
    blue_channel={"label": "Blue Channel", "choices": list(range(1, layer_nb+1))},
    red_channel={"label": "Red Channel", "choices": list(range(1, layer_nb+1))},	
    cropping= {"label": "Crop", "choices": ['No', 'Manual', 'Auto']},	
    start_stack={"label": "Start Z-Stack", "widget_type": "SpinBox", "min": 0, "max": stack_nb, "step": 1, "value": 0, "visible": False},
    end_stack={"label": "End Z-Stack", "widget_type": "SpinBox", "min": 0, "max": stack_nb, "step": 1, "value": stack_nb, "visible": False},
    call_button="Run"
)
    def parameter_selector(blue_channel: int, red_channel: int, cropping, start_stack: int, end_stack: int):
        """Store selected channels and stack range."""
        if blue_channel == red_channel:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please choose different channels.")
            msg.setWindowTitle("Channel Selection Error")
            msg.exec_()
            return
        
        if start_stack == end_stack:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please choose a valid stack range.")
            msg.setWindowTitle("Stack Range Error")
            msg.exec_()
            return

        if start_stack > end_stack:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please choose a valid stack range.")
            msg.setWindowTitle("Stack Range Error")
            msg.exec_()
            return
        
        selected_channels["blue_channel"] = blue_channel - 1
        selected_channels["red_channel"] = red_channel - 1
        cropping_method["cropping"] = cropping
        stack_range["start"] = start_stack+1
        stack_range["end"] = end_stack
        event_loop.quit()  # Quit event loop after selection

    @parameter_selector.cropping.changed.connect
    def _on_mode_change(value: str):
        parameter_selector.start_stack.visible = (value == "Manual")
        parameter_selector.end_stack.visible = (value == "Manual")


    # Add real-time update handlers
    @parameter_selector.blue_channel.changed.connect
    @parameter_selector.red_channel.changed.connect
    def _update_channels(value: int):
        blue_channel = (parameter_selector.blue_channel.value - 1)
        red_channel = (parameter_selector.red_channel.value - 1)
        for i, layer in enumerate(viewer.layers):
            #change the color of the selected channels to blue and red according to the user's selection
            if i == blue_channel:
                layer.colormap = 'cyan'
            elif i == red_channel:
                layer.colormap = 'magenta'
            else:
                layer.colormap = 'grey'

        if parameter_selector.cropping.value == "Auto":
            if blue_channel == red_channel:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Please choose different channels.")
                msg.setWindowTitle("Channel Selection Error")
                msg.exec_()

                ## Reset the cropping method to "No" if the channels are the same
                parameter_selector.cropping.value = "No"
                return
            
    @parameter_selector.cropping.changed.connect
    @parameter_selector.start_stack.changed.connect
    @parameter_selector.end_stack.changed.connect
    def _update_stack_range(value: int):
        if parameter_selector.cropping.value == "Manual":
            start = parameter_selector.start_stack.value
            end = parameter_selector.end_stack.value
            #when the user starts moving the slider, show the selected range of stacks in the mask layer
            #the stacks between 0 and start will be white, the stacks between start and end will be black and the stacks between end and the last stack will be white
        
        if parameter_selector.cropping.value == "Auto":
            blue_channel = (parameter_selector.blue_channel.value - 1)
            red_channel = (parameter_selector.red_channel.value - 1)
            if blue_channel == red_channel:
                print(blue_channel, red_channel)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Please choose different channels.")
                msg.setWindowTitle("Channel Selection Error")
                msg.exec_()

                ## Reset the cropping method to "No" if the channels are the same
                parameter_selector.cropping.value = "No"

                return
            
            start, end = auto_crop(data, blue_channel, red_channel, image_format)

        if parameter_selector.cropping.value == "No":
            crop_mask = np.zeros_like(viewer.layers[-1].data)
            viewer.layers[-1].data = crop_mask

        else:
            if image_format["sizeZ"] > 1:
                crop_mask = np.zeros_like(viewer.layers[-1].data)
                crop_mask[start, :, :] = 1
                crop_mask[end - 1, :, :] = 1
                viewer.layers[-1].data = crop_mask
    
    # Close event: Stop if user closes the window manually
    def on_close(event):

        event_loop.quit()  # Quit event loop if user manually closes Napari
    
    viewer.window._qt_window.closeEvent = on_close  # Capture close event

    parameter_selector.native.setMinimumWidth(400)
    # Add widgets to the Napari viewer
    viewer.window.add_dock_widget(parameter_selector, area="right")

    # Run Napari and block execution
    viewer.window._qt_window.show()
    event_loop.exec_()
    
    event_loop.quit()  # Ensure event loop quits fully before cleanup

    # After exiting the event loop, close the viewer
    viewer.close()

    return selected_channels["blue_channel"], selected_channels["red_channel"], stack_range["start"], stack_range["end"], cropping_method["cropping"]

def auto_crop(data, blue_channel, red_channel, image_format):
    if image_format["sizeT"] > 1 and image_format["sizeZ"] > 1:
        blue_layer = data[:, blue_channel, :, :, :]
        red_layer = data[:, red_channel, :, :, :]

    
    elif image_format["sizeT"] == 1 and image_format["sizeZ"] > 1:
        blue_layer = data[blue_channel, :, :, :]
        red_layer = data[red_channel, :, :, :]
    
    elif image_format["sizeT"] > 1 and image_format["sizeZ"] == 1:
        blue_layer = data[:, blue_channel, :, :]
        red_layer = data[:, red_channel, :, :]

    sum_layer = blue_layer + red_layer

    #make a bar plot with intensities of sum_layer_z
    var_layer_zx = np.var(sum_layer, axis=1)
    var_layer_z = np.var(var_layer_zx, axis=1)

    start_stack = np.argmax(var_layer_z > (np.max(var_layer_z) /5))
    #set end_stack to the last index of var_layer_z
    end_stack = image_format["sizeZ"]

    return start_stack, end_stack

def channel_stack_crop(data, image_format, blue_channel=None, red_channel=None, cropping_method=None, start_stack=None, end_stack=None):
    """Crop the selected channels and stack range."""
    if blue_channel is None or red_channel is None or cropping_method is None:
        blue_channel, red_channel, start_stack, end_stack, cropping_method = specify_channel_crop(data, image_format)

    if cropping_method == "Auto":
        start_stack, end_stack = auto_crop(data, blue_channel, red_channel, image_format)

    elif cropping_method == "No":
        start_stack = 0
        end_stack = image_format["sizeZ"]

    if blue_channel is None or red_channel is None or (cropping_method == "Manual" and (start_stack is None or end_stack is None)):
        raise SystemExit

    if image_format["sizeT"] > 1 and image_format["sizeZ"] > 1:
        data[:, :, :start_stack+1, :, :] = 0
        data[:, :, end_stack:, :, :] = 0
        blue_layer = data[:, blue_channel, :, :, :]
        red_layer = data[:, red_channel, :, :, :]
        additional_layers = [data[:, i, :, :, :] for i in range(image_format["sizeC"]) if i not in [blue_channel, red_channel]]


    
    elif image_format["sizeT"] == 1 and image_format["sizeZ"] > 1:
        data[:, :start_stack+1, :, :] = 0
        data[:, end_stack:, :, :] = 0
        blue_layer = data[blue_channel, :, :, :]
        red_layer = data[red_channel, :, :, :]
        additional_layers = [data[i, :, :, :] for i in range(image_format["sizeC"]) if i not in [blue_channel, red_channel]]

    
    elif image_format["sizeT"] > 1 and image_format["sizeZ"] == 1:
        blue_layer = data[:, blue_channel, :, :]
        red_layer = data[:, red_channel, :, :]
        additional_layers = [data[:, i, :, :] for i in range(image_format["sizeC"]) if i not in [blue_channel, red_channel]]

    return blue_layer, red_layer, start_stack, end_stack, additional_layers, blue_channel, red_channel, cropping_method

