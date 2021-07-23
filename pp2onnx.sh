paddle2onnx \
    --model_dir inference/rec/mv3_none_none_ctc \
    --model_filename inference.pdmodel\
    --params_filename inference.pdiparams \
    --save_file mv2_v2.onnx \
    --opset_version 12

