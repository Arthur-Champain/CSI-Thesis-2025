from os import path
from GP_modules.file_import import load_czi_file, select_path_list
from GP_modules.file_preprocess import channel_stack_crop
from GP_modules.GP_calc import GP_calculation, GP_calculation_chunked
from GP_modules.image_analysis import object_labelling, order_threshold_selector, final_visualization, save_results 


#ask for results folder
#get list of output folders
#if time series -> take frame 1, else get global stats
#segmentation for multi_GPMVs files
#output = violin plot



#ask for list of input folder with czi files
#get parent folder of the input folders -> results folder
#normal process for first file, then remember input parameters and apply them without choice for all other files
#important to delete variables betrween 2 iterations
#separate main function and workflow function 
#threshold out bottom slide

def workflow(output_paths, file_path, blue_channel = None, red_channel = None, cropping_method = None, start_stack = None, end_stack = None, sum_threshold = None, order_threshold = None):
    
    # Define the file path

    #file_path = r"C:\Users\champ\Documents\Master_Thesis_CSI\data\images\250131_GPMVs_RPE-1_4mM-DTT_Media_25nM-NR12A-HBSS\250131_GPMVs_RPE-1_4mM-DTT_HBSS_500nM-NR12S-HBSS_stack_02_BP570-610_LP655-Channel Alignment-35-Lattice Lightsheet-57.czi"
    
    data, image_format, file_path = load_czi_file(file_path)
    if not file_path:
        raise SystemExit
    
    # Get base directory and filename
    base_path = path.dirname(file_path)
    filename = path.splitext(path.basename(file_path))[0]

    
    #print(blue_channel, red_channel, start_stack, end_stack, sum_threshold, order_threshold)
    
    
    blue_layer, red_layer, start_stack, end_stack, additional_layers, blue_channel, red_channel, cropping_method = channel_stack_crop(data, image_format, blue_channel, red_channel, cropping_method, start_stack, end_stack)
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

    min_val, max_val = save_results(output_paths, filename, GP_layer, image_format, sum_threshold, order_threshold, all_object_labels, all_object_props)
    print('Results saved')

    #order ratio = ordered pixels / (ordered pixels + disordered pixels)

    return blue_channel, red_channel, cropping_method, start_stack, end_stack, sum_threshold, order_threshold

def main():

    try:

        file_list = select_path_list()
        for c, file_path in enumerate(file_list):
            print(f'Processing file {c+1} out of {len(file_list)}')
            if c == 0:
                output_paths = path.join(path.dirname(file_path), 'Distribution_Results')
                blue_channel, red_channel, cropping_method, start_stack, end_stack, sum_threshold, order_threshold = workflow(output_paths, file_path)

            else:
                blue_channel, red_channel, cropping_method, start_stack, end_stack, sum_threshold, order_threshold = workflow(output_paths, file_path, blue_channel, red_channel, cropping_method, start_stack, end_stack, sum_threshold, order_threshold)
    
    except SystemExit:
        print("\nExiting the script at the user's request.\n")
        return

if __name__ == "__main__":
    main()