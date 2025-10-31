import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def plot_ppg_signals(predictions, labels, config, filename_id, max_samples=5000):
    """Plot predicted PPG vs ground truth PPG signals and save to file."""
    # Define save path - consistent with BlandAltman plot directory structure
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
        save_path = os.path.join(config.LOG.TEST_PATH, config.TEST.DATA.EXP_DATA_NAME, 'ppg_plots')
    elif config.TOOLBOX_MODE == 'unsupervised_method':
        save_path = os.path.join(config.LOG.TEST_PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'ppg_plots')
    else:
        raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')
    
    # Make the save path, if needed
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    print(f"\nGenerating PPG plots for {len(predictions)} videos...")
    
    # Iterate through each video (subject)
    for video_idx, video_id in enumerate(predictions.keys()):
        # Create a folder for this video
        video_folder = os.path.join(base_save_path, str(video_id))
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
        
        # Get all chunks for this video (sorted by chunk index)
        video_predictions = predictions[video_id]
        video_labels = labels[video_id]
        
        # Sort chunks by their index
        sorted_chunk_indices = sorted(video_predictions.keys())
        
        print(f"  Processing video {video_id} ({video_idx + 1}/{len(predictions)}): {len(sorted_chunk_indices)} chunks")
        
        # Plot each chunk
        for chunk_idx in sorted_chunk_indices:
            prediction = video_predictions[chunk_idx].cpu().numpy().flatten()
            label = video_labels[chunk_idx].cpu().numpy().flatten()
            
            # Create time axis (in seconds if FS is available)
            if hasattr(config.TEST.DATA, 'FS') and config.TEST.DATA.FS:
                time_axis = np.arange(len(prediction)) / config.TEST.DATA.FS
                x_label = 'Time (seconds)'
            else:
                time_axis = np.arange(len(prediction))
                x_label = 'Sample Index'
            
            # Create the plot for this chunk
            plt.figure(figsize=(15, 6))
            plt.plot(time_axis, label, label='Ground Truth PPG', alpha=0.7, linewidth=1)
            plt.plot(time_axis, prediction, label='Predicted PPG', alpha=0.7, linewidth=1)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel('PPG Signal', fontsize=12)
            plt.title(f'Video {video_id} - Chunk {chunk_idx} - Predicted vs Ground Truth PPG', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            file_name = f'chunk_{chunk_idx}_PPG_comparison.pdf'
            save_file = os.path.join(video_folder, file_name)
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"    Saved {len(sorted_chunk_indices)} chunk plots to: {video_folder}")
    
    print(f"Completed PPG plotting for all {len(predictions)} videos!")


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()

    predict_hr_fft_all_per_vid = list()
    gt_hr_fft_all_per_vid = list()
    predict_hr_peak_all_per_vid = list()
    gt_hr_peak_all_per_vid = list()
    SNR_all_per_vid = list()
    MACC_all_per_vid = list()
    print("Calculating metrics!")
    per_video_results = []
    chunk_results = []

    pred_ppg_all = []
    gt_ppg_all = []

    # predictions.keys() is the name of the video chunks
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        pred_ppg_all.extend(label.tolist())
        gt_ppg_all.extend(prediction.tolist())

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size
        

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)

                gt_hr_fft_all_per_vid.append(gt_hr_fft)
                predict_hr_fft_all_per_vid.append(pred_hr_fft)
                SNR_all_per_vid.append(SNR)
                MACC_all_per_vid.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")

            chunk_results.append({
                "video_id": index,
                "chunk_index": i // window_frame_size,
                "gt_hr": gt_hr_fft,
                "pred_hr": pred_hr_fft,
                "SNR": SNR,
                "MACC": macc
            })

        per_video_results.append({
            "video_id": index,
            "avg_gt_hr_fft": np.mean(gt_hr_fft_all_per_vid),
            "avg_pred_hr_fft": np.mean(predict_hr_fft_all_per_vid),
            "avg_SNR": np.mean(SNR_all_per_vid),
            "avg_MACC": np.mean(MACC_all_per_vid),
        })

        gt_hr_peak_all_per_vid.clear()
        predict_hr_peak_all_per_vid.clear()
        gt_hr_fft_all_per_vid.clear()
        predict_hr_fft_all_per_vid.clear()
        SNR_all_per_vid.clear()
        MACC_all_per_vid.clear()

    # Ensure the output directory exists before saving CSV files
    if not os.path.exists(config.TEST.OUTPUT_SAVE_DIR):
        os.makedirs(config.TEST.OUTPUT_SAVE_DIR, exist_ok=True)
    
    all_chunk_results_df = pd.DataFrame(chunk_results)
    chunk_csv_path = os.path.join(config.TEST.OUTPUT_SAVE_DIR, config.TRAIN.MODEL_FILE_NAME + "_per_chunk_metrics.csv")
    all_chunk_results_df.to_csv(chunk_csv_path, index=False)
    print(f"[INFO] Saved per-chunk metrics to: {chunk_csv_path}")

    df = pd.DataFrame(per_video_results)
    csv_path = os.path.join(config.TEST.OUTPUT_SAVE_DIR, config.TRAIN.MODEL_FILE_NAME + "_per_video_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved per-video metrics to: {csv_path}")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            # If BA is in yaml file, the bland altman plot will be saved
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
    
    # Plot predicted PPG vs ground truth PPG signals
    print("\nGenerating PPG signal comparison plots...")
    plot_ppg_signals(predictions, labels, config, filename_id)
