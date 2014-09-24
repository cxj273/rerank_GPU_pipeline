#!/usr/bin/env python

import sys
import os

def main(argv):
    if len(argv) != 14:
        print "Usage: ./rerank_wrapper.py all_feat_list dev_feat evl_mapped_feat original_kernel original_label PQ_codebook event_name rerank_feat_config top_n dev_num gpu_id EKn('EK10' or 'EK100') cross_validation_folder_num evl_label('None' for no label)"
        sys.exit(1)

    all_feat_list_path = argv[0]
    dev_feat_path = argv[1]
    evl_mapped_feat_path = argv[2]
    original_kernel_path = argv[3]
    original_label_path = argv[4]
    PQ_codebook_path = argv[5]
    event_name = argv[6]
    rerank_feat_config_path = argv[7]
    top_n = argv[8]
    dev_num = argv[9]
    gpu_id = argv[10]
    ek_n = argv[11]
    cross_folder_num = argv[12]
    evl_label_path = argv[13]

    event_folder = os.getcwd() + "/" + event_name

    new_kernel_path = "/".join([event_folder, event_name + ".new_dev.dist"])
    new_model_path = "/".join([event_folder, event_name + ".new_model"])
    new_label_path = "/".join([event_folder, event_name + ".new_dev.label"])
    #model_config_path = "/".join([event_folder, event_name + ".model.config"])
    #model_script_path = "/".join([event_folder, event_name + ".model.m"])
    #model_config = " ".join([event_name, "Default", new_label_path ])
    #model_script = '\n'.join([r"addpath('/home/iyu/MyCodes/lab/MED14_pipeline/linear_regression/');", "main("+ ', '.join(["'" + model_config_path + "'","'" + new_kernel_path + "'", dev_num, "'" + new_model_path + "'", cross_folder_num, "'" + ek_n + "'"])+");", "exit;"])

    top_feat_list_path = "/".join([event_folder, event_name + ".rerank_feat.list"])
    top_feat_idx_path = "/".join([event_folder, event_name + ".rerank_feat.idx"])
    top_feat_label_path = "/".join([event_folder, event_name + ".rerank_feat.label"])

    rerank_kernel_bin = "./bin/rerank_kernel"
    rerank_train_bin = "./bin/rerank_train"
    libsvm2linear_bin = "./bin/libsvm2linear_rerank"
    PQ_model_bin = "./bin/PQ_model"
    PQ_predict_bin = "./bin/PQ_predict_gpu"
    ap_bin = "./bin/ap.py"
    select_files_bin = "./bin/select_files.pl"
    led_bin = "./bin/led.py"

    libsvm_model_path = new_model_path + "/" + event_name + ".model"
    linear_model_path = new_model_path + "/" + event_name + ".linear_model"
    PQ_model_path = new_model_path + "/" + event_name + ".PQ_model"
    prediction_path = new_model_path + "/" + event_name + ".prediction"
    ap_path = new_model_path + "/" + event_name + ".ap"


    cd_cmd = "cd " + os.getcwd()
    mkdir_cmd = "mkdir " + event_name
    set_folder_cmd = " && ".join([cd_cmd, mkdir_cmd])

    print "Setting folder: " + set_folder_cmd
    os.system(set_folder_cmd)

    all_feat_dict = dict()
    with open( all_feat_list_path, "r") as all_feat_list:
        for path in all_feat_list:
            key = path.split('/')[-1].split('.')[0]
            all_feat_dict[key.lstrip("HVC")]=path

    top_n_int = 0
    with open( rerank_feat_config_path , "r") as top_feat_idx: 
        with open( top_feat_idx_path, "w") as top_feat_idx_file:
            with open( top_feat_label_path, "w") as top_feat_label_file: 
                for line in top_feat_idx: 
                    label = line.split()[1] 
                    idx = line.split()[0] 
                    if idx in all_feat_dict: 
                        top_n_int += 1
                        top_feat_idx_file.write(idx+"\n")
                        top_feat_label_file.write(label+"\n")

    top_n = str(top_n_int) 
    print top_n

    parse_top_feat_idx_cmd = led_bin + " " + rerank_feat_config_path + " \"line.split()[0]\" > " + top_feat_idx_path
    parse_top_feat_label_cmd = led_bin + " " + rerank_feat_config_path + " \"line.split()[1]\" > " + top_feat_label_path
    select_files_cmd = "perl " + select_files_bin + " " + all_feat_list_path + " " + top_feat_idx_path + " > " + top_feat_list_path
    generate_new_label_cmd = "cat " + original_label_path + " " + top_feat_label_path + " > " + new_label_path

    parse_rerank_feat_config_cmd = " && ".join([select_files_cmd,generate_new_label_cmd])
    print "Finding feature and Creating new label: " + parse_rerank_feat_config_cmd
    os.system(parse_rerank_feat_config_cmd)


    libsvm2linear_cmd = " ".join([libsvm2linear_bin, libsvm_model_path, dev_feat_path, linear_model_path, top_feat_list_path, top_n])
    PQ_model_cmd = " ".join([PQ_model_bin, linear_model_path, PQ_codebook_path, PQ_model_path])
    PQ_predict_cmd = " ".join([PQ_predict_bin, evl_mapped_feat_path, PQ_model_path, prediction_path, gpu_id])
    
    rerank_train_cmd = " ".join([rerank_train_bin, dev_feat_path, original_kernel_path, top_feat_list_path, top_n, ek_n, cross_folder_num, "6", new_label_path, libsvm_model_path, gpu_id])
    ap_cmd = " ".join([ap_bin, prediction_path, evl_label_path]) + " > " + ap_path
    mkdir_new_model_cmd = "mkdir " + new_model_path

    gen_kernel_cmd = " && ".join([cd_cmd, mkdir_new_model_cmd, rerank_train_cmd])
    print "Generating Kernel and Training: " + gen_kernel_cmd
    os.system(gen_kernel_cmd)

    #with open(model_config_path, "w") as config_file:
    #    config_file.write(model_config)

    #with open(model_script_path, "w") as script_file:
    #    script_file.write(model_script)

    #train_cmd = "matlab -nodisplay < " + model_script_path
    #
    #print "Training model: " + train_cmd
    #os.system(train_cmd)
    predict_cmd = " && ".join([libsvm2linear_cmd, PQ_model_cmd, PQ_predict_cmd])
    print "Prediction: " + predict_cmd 
    os.system(predict_cmd)

    if evl_label_path != "None":
        print "Computing AP: " + ap_cmd
        os.system(ap_cmd)

if __name__ == "__main__":
    main(sys.argv[1:])
