# CS2225.CH1507 - Nhóm 7 - Nhận dạng trái cây

[CH2001010	Mai Phương Nga](mailto:ngamp.15@gm.uit.edu.vn?subject=[GitHub]%CS2225.CH1507%Nhan%dang%trai%cay)

[CH2001015	Nguyễn Như Thanh](mailto:thanhnn.15@gm.uit.edu.vn?subject=[GitHub]%CS2225.CH1507%Nhan%dang%trai%cay)  

[CH2001001	Trần Hiếu Đại](mailto:daith.15@gm.uit.edu.vn?subject=[GitHub]%CS2225.CH1507%Nhan%dang%trai%cay)

## Giới thiệu

1. Tên đề tài: Nhận dạng trái cây dựa trên hình ảnh

2. Mô tả đề tài: Xây dựng một hệ thống giúp nhận dạng trái cây của VN dựa trên ảnh chụp.

<p align="center">
 <img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/System.2.jpg" width="75%" height="75%">
</p>

Trong đó:
* Input là 1 ảnh chỉ chứa 1 hoặc nhiều trái cây trong 6 loại sau đây: **`thanh long, măng cụt, mận, ổi, xoài, khế`**
* Hệ thống Nhận dạng trái cây sẽ xử lý và detect trái cây có trên ảnh chụp được đưa vào
* Output là bounding box cho từng vùng có trái cây và thông tin loại trái cây tương ứng mà hệ thống detect được

3. Công cụ và thư viện: Colab, python 3.x, Tensorflow version 2, Object Detection Api 

## Video Demo

[![Demo](https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/Intro.jpg)](https://youtu.be/hhwftzrl_CQ)

## Các bản cập nhật

**`2021-02-07`**: Cập nhật pretrained model và notebook để testing với dataset resize 416x416, kèm thêm augmentation. Xem thêm [source code](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/source_code)

**`2021-02-05`**: Cập nhật pretrained model và notebook để testing với dataset resize 226x226, kèm thêm augmentation. Xem thêm [v2_226x226_noise_bright_grayscale](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v2_226x226_noise_bright_grayscale)

**`2021-02-04`**: Update pretrained model và tạo notebook để testing với dataset resize 150x150. Xem thêm [v2_150x150_raw](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v2_150x150_raw)

**`2021-02-01`**: Cập nhật dataset, tăng thêm độ đa dạng của các ảnh chụp. Xem dataset mới nhất [tại đây](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/dataset/dataset_with_annotation).  

**`2020-12-14`**: [Hệ thống](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v1_240x240_noise_bright) dùng dataset gồm các ảnh chụp đơn giản của từng loại trái cây. 

## Nhận dạng trái cây

### Giới thiệu

### Training data

Training data được nhóm thu thập qua ảnh chụp trực tiếp từ điện thoại và nguồn ảnh trên Internet. Quá trình gán nhãn được thực hiện thủ công. Tập ảnh đã được upload lên [Roboflow](https://app.roboflow.com/ds/6kyOg1KHvY?key=9NoENEKLqj) để tiện xử lý. 

[Dataset](https://github.com/MaiNga-uit/CS2225.CH1507) bao gồm:

* Tập train: 1566 hình được generate dựa trên 512 hình (80% dataset) kèm thêm các bước tiền xử lý và gia tẳng bộ ảnh, bao gồm: resize - 416x416 (fit white background); rotation: -45 độ và +45 độ; shear: +-15 Horizontal, +-15 Vertical; brightness: +-20%; blur: up to 5px; noise: up to 5%
* Tập test: 130 hình (20%)

### Train

Các thông tin sau đây mô tả một cách khái quát các bước cần thực hiện để training. Chi tiết có thể tham khảo tại [Notebook](https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/source_code/%5BCS2225_CH1501%5D6_fruits_object_detection.ipynb)

Bước 1. Clone the tensorflow models repository từ github về

```
!git clone --depth 1 https://github.com/tensorflow/models
```

Bước 2. Cài đặt Object Detection API và import những thư viện cần thiết

```
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install 
```
```
!python /content/models/research/object_detection/builders/model_builder_tf2_test.py
```

Bước 3. Download dataset của nhóm từ Roboflow

```
%cd /content
!curl -L "https://app.roboflow.com/ds/6kyOg1KHvY?key=9NoENEKLqj" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

Bước 5. Những config cần thiết trước khi build model

* Config path: test_record_fname, train_record_fname, label_map_pbtxt_fname
* Mount drive để lưu file model sau khi build
* Config num_steps, hiện tại chỉnh về 2000 step.

Bước 6. Train model

```
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}
```

```
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --checkpoint_dir={model_dir}
```

Bước 7. Testing

```
detections, predictions_dict, shapes = detect_fn(input_tensor)
```

```
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.8,
      agnostic_mode=False,
      line_thickness=3
)
```

<img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/testing.png">

### Evaluation

Kết quả đánh giá dựa trên Mean Average Precision và Average Recall

<img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/evaluation/Eval.mAP.jpg">

<img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/evaluation/Eval.AR.jpg">

