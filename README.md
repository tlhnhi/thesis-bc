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
