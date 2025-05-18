from os import path
from GP_modules.file_import import load_czi_file
from GP_modules.file_preprocess import channel_stack_crop
from GP_modules.GP_calc import GP_calculation, GP_calculation_chunked
from GP_modules.image_analysis import object_labelling, order_threshold_selector, final_visualization, save_results

def main(blue_channel = None, red_channel = None, cropping_method = None, sum_threshold = None, order_threshold = None):
    # Define the file path
    try:        
        data, image_format, file_path = load_czi_file()
        if not file_path:
            raise SystemExit
        
        # Get base directory and filename
        save_path = path.dirname(file_path)
        save_path = path.join(save_path, 'Results')
        filename = path.splitext(path.basename(file_path))[0]
        
        blue_layer, red_layer, start_stack, end_stack, additional_layers, blue_channel, red_channel, cropping_method = channel_stack_crop(data, image_format)

        print('Image croppped and channel specified')

        del data  # Free up memory

        #GP_layer, sum_threshold = GP_calculation(blue_layer, red_layer, image_format)
        GP_layer, sum_threshold = GP_calculation_chunked(blue_layer, red_layer, image_format, sum_threshold)
        print('GP calculation done')

        del blue_layer, red_layer, additional_layers  # Free up memory

        all_object_labels, all_object_props  =  object_labelling(GP_layer, image_format)
        print('Object labelling done')

        if order_threshold is None:
            order_threshold = order_threshold_selector(GP_layer, image_format)
            print('Order threshold selected')

        if order_threshold is None:
            raise SystemExit

        min_val, max_val = save_results(save_path, filename, GP_layer, image_format, sum_threshold, order_threshold, all_object_labels, all_object_props)
        print('Results saved')

        final_visualization(GP_layer, 
                            order_threshold,
                            image_format,
                            all_object_labels, 
                            min_val,
                            max_val)
        
        print('Visualization done')
    
    except SystemExit:
        print("\nExiting the script at the user's request.\n")
        return
    
if __name__ == "__main__":
    main()