# Demo Khóa luận tốt nghiệp - K17 - HCMUS

Đây là ứng dụng web minh họa khả năng hoạt động của kết quả được trình bày trong khóa luận đề tài _Áp dụng kỹ thuật học sâu hỗ trợ phát hiện và phân loại tế bào máu theo hình dạng_

## Sinh viên thực hiện

| Student ID | Name             |
| ---------- | ---------------- |
| 1753081    | Trần Lê Hồng Nhi |
| 1753083    | Nguyễn Hưng Phát |

## Môi trường

Web minh họa được phát triển bằng Python 3.9 với các thư viện hỗ trợ như sau:

- **Pytorch**

(_Cài đặt theo hướng dẫn của Pytorch [tại đây](https://pytorch.org/get-started/locally/)_)

- **Detectron2**

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

- **Flask**

```
pip install Flask
```

## Cách sử dụng

- Clone repo này về máy và di chuyển vào thư mục repo
- Tải folder `weights` [tại đây](https://drive.google.com/drive/folders/16ysq-YQ4MJUHRfTvMHTj9TQhaKhn-hRJ) và chuyển vào thư mục chính của repo

```
- thesis-bc\
|---- statis\
|---- templates\
|---- weights\
|---- README.md
|---- demo_app.py
```

- Export biến môi trường `FLASK_APP` (1 trong 3 cách)
  - Với Bash
  ```
  export FLASK_APP=demo_app.py
  ```
  - Với CMD
  ```
  set FLASK_APP=demo_app.py
  ```
  - Với Powershell
  ```
  $env:FLASK_APP = "demo_app.py"
  ```
- Chạy server

```
flask run
```

- Sau khi chờ model được tải lên, server sẽ chạy tại http://127.0.0.1:5000/

## Huấn luyện mô hình

### Phát hiện tế bào

Các mô hình huấn luyện tế bào đều được huấn luyện trên Google Colab

- [Faster R-CNN](https://colab.research.google.com/drive/1s7K4f-xvlZ96mX8glSpgLQ1GQdH7ml1i)
- [YOLOv5](https://colab.research.google.com/drive/14-8lng9oDvBX3hXiGEyfNgIq0OgrodD5)

**Note**: Các dữ liệu đều được augment bằng [Roboflow](https://app.roboflow.com/) nên token tải dataset về có thể hết hạn

### Phân loại tế bào

Bố trí thư mục cho tập dữ liệu bao gồm 2 thư mục con là `train` (dùng cho huấn luyện) và `valid` (dùng cho đánh giá), trong đó mỗi thư mục con chứa các thư mục nhỏ hơn là thư mục hình ảnh tương ứng với phân loại của tập dữ liệu, VD:

```
|---- data\
      |---- train\
            |---- basophil\
            |---- eosinophil\
            |---- lymphocyte\
            |---- monocyte\
            |---- neutrophil\
      |---- valid\
            |---- basophil\
            |---- eosinophil\
            |---- lymphocyte\
            |---- monocyte\
            |---- neutrophil\

```

Huấn luyện bằng lệnh

```
python3 train.py [--model MODEL] [--dataset DATASET] [--classes CLASSES]
                 [--batch BATCH] [--epochs EPOCHS] [--device DEVICE]

```

Trong đó,

- `--model MODEL`: Tên mô hình (resnet, alexnet, vgg, squeezenet, densenet)
- `--dataset DATASET`: Đường dẫn đến tập dữ liệu
- `--classes CLASSES`: Số lớp huấn luyện (`3` đối với hồng cầu và `5` đối với bạch cầu)
- `--batch BATCH`: Batch-size huấn luyện
- `--epochs EPOCHS`: Số epochs huấn luyện
- `--device DEVICE`: `"cpu"` hoặc `0`, `1`, ...

#### Đánh giá với k-fold

Tập dữ liệu huấn luyện được bố trí là một thư mục gồm các thư mục con chứa hình ảnh tương ứng với các phân loại của tập dữ liệu, VD:

```

|---- train\
      |---- basophil\
      |---- eosinophil\
      |---- lymphocyte\
      |---- monocyte\
      |---- neutrophil\

```

Cài đặt thêm các thư viện hổ trợ

- **scikit-learn**

```
pip install scikit-learn
```

- **Matplotlib**

```
pip install matplotlib
```

Huấn luyện bằng lệnh

```
python3 train.py [--model MODEL] [--dataset DATASET] [--classes CLASSES] [--k-folds K_FOLDS]
                 [--batch BATCH] [--epochs EPOCHS] [--device DEVICE]

```

Trong đó,

- `--model MODEL`: Tên mô hình (resnet, alexnet, vgg, squeezenet, densenet)
- `--dataset DATASET`: Đường dẫn đến tập dữ liệu
- `--classes CLASSES`: Số lớp huấn luyện (`3` đối với hồng cầu và `5` đối với bạch cầu)
- `--k-folds K_FOLDS`: Giá trị của k
- `--batch BATCH`: Batch-size huấn luyện
- `--epochs EPOCHS`: Số epochs huấn luyện
- `--device DEVICE`: `"cpu"` hoặc `0`, `1`, ...
