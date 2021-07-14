# recommended paddle.__version__ == 2.0.0
###### 训练 ######
# rec_chinese_lite_train
#python3 -m paddle.distributed.launch \
#  --log_dir=./debug/ \
#  --gpus '0' \
#  tools/train.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml

# rec_mv3_none_none_ctc
python3 -m paddle.distributed.launch \
  --log_dir=./debug/ \
  --gpus '0' \
  tools/train.py \
  -c configs/custom/rec_mv3_none_none_ctc.yml

# rec_r34_none_none_ctc
#python3 -m paddle.distributed.launch \
#  --log_dir=./debug/ \
#  --gpus '0' \
#  tools/train.py \
#  -c configs/custom/rec_r34_vd_none_none_ctc.yml

###### rec模型 直接预测 ######
#python3 tools/infer_rec.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml \
#  -o Global.pretrained_model=output/rec/rec_chinese_lite_v2.0/best_accuracy \
#  Global.load_static_weights=false \
#  Global.infer_img=/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/

#python3 tools/infer_rec.py \
#  -c configs/custom/rec_mv3_none_none_ctc.yml \
#  -o Global.pretrained_model=output/rec/mv3_none_none_ctc/best_accuracy \
#  Global.load_static_weights=false \
#  Global.infer_img=/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/
#
#python3 tools/infer_rec.py \
#  -c configs/custom/rec_r34_none_none_ctc.yml \
#  -o Global.pretrained_model=output/rec/r34_none_none_ctc/best_accuracy \
#  Global.load_static_weights=false \
#  Global.infer_img=/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/

###### rec 模型转化为 infer模型 ######
#python3 tools/export_model.py \
#  -c configs/custom/rec_chinese_lite_train_v2.0.yml \
#  -o Global.pretrained_model=output/rec/rec_chinese_lite_v2.0/best_accuracy \
#  Global.save_inference_dir=./inference/rec/rec_chinese_lite_v2

#python3 tools/export_model.py \
#  -c configs/custom/rec_mv3_none_none_ctc.yml \
#  -o Global.pretrained_model=output/rec/mv3_none_none_ctc/best_accuracy \
#  Global.save_inference_dir=./inference/rec/mv3_none_none_ctc

#  --image_dir="/sdata/dataset/chineseocr-lite/line/1/" \
###### 使用infer模型进行预测 ######
#python3 tools/yjy_infer/predict_rec.py \
#  --image_dir="/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/" \
#  --rec_model_dir="./inference/rec/rec_chinese_lite_v2/" \
#  --rec_image_shape='3,32,512' \
#  --rec_char_type='ch' \
#  --rec_char_dict_path="ppocr/utils/dict/char_std_v1.txt" \
#  --rec_batch_num=1 \
#  --use_gpu=False \
#  --rec_pre_save_dir="/sdata/airesult/ppocr/rec/rec_chinese_lite_v2.0/20210714_v2"

#python3 tools/yjy_infer/predict_rec.py \
#  --image_dir="/sdata/dataset/Synthetic_Chinese_String_Dataset/test_data/0/" \
#  --rec_model_dir="./inference/rec/mv3_none_none_ctc/" \
#  --rec_image_shape='3,32,280' \
#  --rec_char_type='ch' \
#  --rec_char_dict_path="ppocr/utils/dict/char_std_v1.txt" \
#  --rec_batch_num=1 \
#  --use_gpu=False \
#  --rec_pre_save_dir="/sdata/airesult/ppocr/rec/mv3_none_none_ctc/20210713"
