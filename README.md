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

Click vào hình dưới để xem

[![Demo](https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/Intro.jpg)](https://youtu.be/hhwftzrl_CQ)

## Các bản cập nhật

**`2021-02-07`**: Cập nhật pretrained model và notebook để testing với dataset resize 416x416, kèm thêm augmentation. Xem thêm [source code](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/source_code)

**`2021-02-05`**: Cập nhật pretrained model và notebook để testing với dataset resize 226x226, kèm thêm augmentation. Xem thêm [v2_226x226_noise_bright_grayscale](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v2_226x226_noise_bright_grayscale)

**`2021-02-04`**: Update pretrained model và tạo notebook để testing với dataset resize 150x150. Xem thêm [v2_150x150_raw](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v2_150x150_raw)

**`2021-02-01`**: Cập nhật dataset, tăng thêm độ đa dạng của các ảnh chụp. Xem dataset mới nhất [tại đây](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/dataset/dataset_with_annotation).  

**`2020-12-14`**: [Hệ thống](https://github.com/MaiNga-uit/CS2225.CH1507/tree/master/old.vers/v1_240x240_noise_bright) dùng dataset gồm các ảnh chụp đơn giản của từng loại trái cây. 

## Nhận dạng trái cây

### Giới thiệu

Sơ lược về bài toán Object detection

* Input: Hình ảnh
* Ouput: Dự đoán vị trí (thể hiện qua bounding box) và tên object.

Lịch sử phát triển: Machine learning truyền thống và Deep learning

1. Machine learning – base

Sử dụng sliding window classifier để trượt trên từng vùng của hình ảnh input từ đó rút trích features để tính toán phân loại. Dựa vào kết quả tính toán để output ra tên object và vị trí object (dựa vào sliding window)

Mô hình machine learning: 

<p align="center">
 <img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/MLPhase.jpg" width="60%" height="60%">
</p>

Việc học theo mô hình machine learning truyền thống đã phát triển trong một thời gian dài và xây dựng nền tảng để giải quyết bài toán object detection cũng như những bài toán khác như classification, localization. Tuy nhiên, việc extract feature tương đối phức tạp, và đòi hỏi 1 số kiến thức chuyên sâu, và chi phí tính toán tương đối cao

2. Deep learning

<p align="center">
 <img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/Deep.jpg" width="60%" height="60%">
</p>

Deep learning dựa trên cấu trúc mạng nơ-ron (tương tự như não người). Mạng nơ-ron sẽ làm nhiệm vụ học trực tiếp từ dữ liệu input đầu vào feature extraction, learning classification,...) để cho ra output. 

Như vậy, thay vì extract feature là 1 task riêng biệt như trong machine learning truyền thống, thì trong deep learning, mạng nơ-ron sẽ làm nhiệm vụ học trực tiếp từ dữ liệu đầu vào (feature extraction, learning classification)

Một số mô hình đang phát triển hiện nay: SSD, YOLO, EfficientDet. Nhóm chúng tôi chọn mô hình EfficcientDet (D0) được hỗ trợ trong Tensorflow (version 2) để giải quyết bài toán “Nhận dạng trái cây” bởi vì:
* Tensorflow là 1 library được Google phát triển, có nhiều tutorial và source code, hỗ trợ nhiều công cụ giúp thời gian cài đặt nhanh.
* EfficientDet D0 là một mô hình nhỏ, có thời gian training nhanh phù hợp với mục địch khởi đầu về nghiên cứu bài toán nhận dạng

Hướng phát triển: nhóm chúng tôi sẽ cài đặt thử nghiệm nhiều mô hình khác nhau như YOLO, để so sánh và đánh giá.

### Training data

Training data được nhóm thu thập qua ảnh chụp trực tiếp từ điện thoại và nguồn ảnh trên Internet. Quá trình gán nhãn được thực hiện thủ công. Tập ảnh đã được upload lên [Roboflow](https://app.roboflow.com/ds/6kyOg1KHvY?key=9NoENEKLqj) để tiện xử lý. 

[Dataset](https://github.com/MaiNga-uit/CS2225.CH1507) bao gồm:

* Tập train: 1566 hình được generate dựa trên 512 hình (80% dataset) kèm thêm các bước tiền xử lý và gia tăng bộ ảnh, bao gồm: resize - 416x416 (fit white background); rotation: -45 độ và +45 độ; shear: +-15 Horizontal, +-15 Vertical; brightness: +-20%; blur: up to 5px; noise: up to 5%
* Tập test: 130 hình (20% dataset)

### Train

Các thông tin sau đây mô tả một cách khái quát các bước cần thực hiện để training. Chi tiết có thể tham khảo tại [Notebook for training](https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/source_code/%5BCS2225_CH1501%5D6_fruits_object_detection.ipynb)

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

### Test

Do quá trình train từ đầu tốn khá nhiều thời gian nên nhóm đã export trước kết quả train. Kết quả này dùng làm input cho Notebook dùng riêng cho việc thử nghiệm. Notebook hỗ trợ test trên một hoặc nhiều ảnh được tải lên, hoặc dựa vào ảnh chụp từ webcam. 

Sau đây là mô tả khái quát về các bước cần thực hiện để testing. Chi tiết có thể tham khảo tại [Notebook for testing](https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/source_code/%5BCS2225%5DTesting_6_fruits_detection_model_with_multiple_images.ipynb)

Bước 1. Download model đã được nhóm train sẵn và upload lên goodle drive

```
import gdown

modelUrl = 'https://drive.google.com/uc?id=12vMCYOzWS9BmZ_iwuGT-8HPKB98glvuS' #URL cố định dùng để download.
output = '/content/trained_model.zip' 
gdown.download(modelUrl, output, quiet=False)

!unzip -o '/content/trained_model.zip' -d '/content/'
!rm -r '/content/trained_model.zip'
```

Bước 2. Cài đặt Object detection API

Bước 3. Import thư viện và config cần thiết trước khi run test

Bước 4. Cách test 1: test bằng cách input 01 hình ảnh

* input 1 hình ảnh, output hiển thị trực tiếp ngay phía dưới đoạn code

Bước 5. Cách test 2: Lấy hình được chụp từ webcam

* Webcam sẽ được bật, click 1 click để chụp hình từ webcam
* Output: hình ảnh được chụp cùng với bounding box, label name, score

Bước 6: Cách test 3: Chạy thử nghiệm trên toàn bộ tập ảnh test và lưu vào drive

* Input: Folder chứa bộ ảnh cần test (kiểu *.jpg)
* Output: Kết quả detect sẽ được ghi vào folder được chỉ định trong Drive

### Evaluation

Kết quả đánh giá dựa trên Mean Average Precision và Average Recall

<img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/evaluation/Eval.mAP.jpg">

<img src="https://github.com/MaiNga-uit/CS2225.CH1507/blob/master/resources/evaluation/Eval.AR.jpg">

Kết quả dựa theo các độ đo trên cho thấy tập dataset được resize về 226x226 kèm các augmentation cho kết quả khả quan nhất. Tuy nhiên trên thực tế, khi nhóm thực hiện kiểm thử với một bộ ảnh validation hoàn toàn độc lập với dataset ban đầu thì configuration trên hoàn toàn không detect được nhãn 'khe', kết quả dự đoán cho ra rất nhiều nhãn 'thanhlong'.

Cũng dựa vào việc thực hiện với tập ảnh validation ở trên, configuration resize 416x416 kèm các augmentation rotate, shear cho ra kết quả khả quan hơn.

### Hướng phát triển

Cần cải thiện hệ thống bằng cách bổ sung thêm dữ liệu đầu vào từ nhiều nguồn khác, ảnh chụp cần đa dạng bối cảnh, bổ sung thêm ảnh chụp có chứa nhiều loại trái cây trong cùng một tấm hình.

Đánh giá với nhiều model và phương pháp khác đang hiện có.

Nhận dạng thời gian thực (real-time detection).

Xử lý thêm dữ liệu đầu vào là video.

### Các nguồn tham khảo

https://blog.roboflow.com/breaking-down-efficientdet

https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api

https://blog.roboflow.com/train-a-tensorflow2-object-detection-model/


Thanks for watching!
