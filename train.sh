# recommended paddle.__version__ == 2.0.0
#训练
#python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
#python3 -m paddle.distributed.launch \
#  --log_dir=./debug/ \
#  --gpus '0' \
#  tools/train.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml

# rec模型 直接预测
#python3 tools/infer_rec.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml \
#  -o Global.pretrained_model=output/rec_chinese_lite_v2.0/best_accuracy \
#  Global.load_static_weights=false \
#  Global.infer_img=/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/

# rec 模型转化为 infer模型
#python3 tools/export_model.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml \
#  -o Global.pretrained_model=output/rec_chinese_lite_v2.0/best_accuracy \
#  Global.save_inference_dir=./inference/rec_crnn/

# 使用infer模型进行预测
python3 tools/infer/predict_rec.py \
  --image_dir="/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/" \
  --rec_model_dir="./inference/rec_crnn/" \
  --rec_image_shape='3,32,280' \
  --rec_char_type='ch' \
  --rec_char_dict_path="ppocr/utils/dict/chinese_cht_dict.txt"
